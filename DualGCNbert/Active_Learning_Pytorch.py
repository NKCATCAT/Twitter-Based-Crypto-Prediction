import os
import sys
import copy
import random
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import json
import pickle
from sklearn import metrics
from time import strftime, localtime
from torch.utils.data import DataLoader
from transformers import BertModel, AdamW
from DualGCNbert.DualGCNBert import DualGCNBertClassifier
from prepare_data import Tokenizer4BertGCN, DualGCNBertData
from collections import defaultdict
from sklearn.model_selection import KFold
import math
from azure.storage.blob import BlobServiceClient
from urllib.parse import quote
#from nltk.tokenize import TweetTokenizer
#tokenizer = TweetTokenizer()
#%%
kf = KFold(n_splits = 2, random_state = 1000, shuffle = True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
'''
logging.StreamHandler is a basic Handler class which writes logging records, 
appropriately formatted, to a stream (defaults to sys.stderr if not specified).sys.stdout means the standard output.
'''
def setup_seed(seed):
    torch.manual_seed(seed) #CPU
    torch.cuda.manual_seed_all(seed)  #GPU
    np.random.seed(seed) 
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
class Instructor:
    ''' Model training and evaluation '''
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4BertGCN(opt.max_length, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt).to(opt.device)
        torch.save(self.model.state_dict(), 'initial_model.pth')
#       self.trainset = DualGCNBertData(opt.dataset_file['initial_en'], tokenizer, opt=opt)
        self.unlabeled_dataset = DualGCNBertData(opt.dataset_file['unlabeled_en'], tokenizer, opt=opt)
        self.unlabeled_dataloader = DataLoader(dataset=self.unlabeled_dataset, batch_size=opt.batch_size)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))
        self.init_model_weights = copy.deepcopy(self.model.state_dict())
        self._print_args()
        
    def _print_args(self):
        '''
        Initialize counters for trainable and non-trainable parameters
        '''
        n_trainable_params, n_nontrainable_params = 0, 0

        for p in self.model.parameters():
            '''
            模型中的每个参数是一个多维度的张量（tensor）
            '''
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
  
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')
        '''
        Iterate over all options
        '''
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
    
    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                '''
                If the parameter tensor has more than one dimension
                '''
                if len(p.shape) > 1:
                    '''
                    Initialize it with torch.nn.init.xavier_uniform_
                    '''
                    self.opt.initializer(p)
                else:
                    '''
                    If the parameter tensor has only one dimension    bias
                    '''
                    stdv = 1. / (p.shape[0]**0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    
    def get_bert_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']        
        diff_part = ["bert.embeddings", "bert.encoder"]
        if self.opt.diff_lr:
            logger.info("layered learning rate on")
            '''
            Define grouped parameters for the optimizer with different learning rates
            '''
            optimizer_grouped_parameters = [
                
                #Parameters for the BERT embeddings and encoder layers that are not bias or LayerNorm weights
                
                {
                    "params": [p for n, p in model.named_parameters() if
                            not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.bert_lr
                },
                
                #Parameters for the BERT embeddings and encoder layers that are bias or LayerNorm weights
        
                {
                    "params": [p for n, p in model.named_parameters() if
                            any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.bert_lr
                },
                
                #Parameters for the non-BERT layers that are not bias or LayerNorm weights
                
                {
                    "params": [p for n, p in model.named_parameters() if
                            not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.learning_rate
                },
                
                #Parameters for the non-BERT layers that are bias or LayerNorm weights
                
                {
                    "params": [p for n, p in model.named_parameters() if
                            any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.learning_rate
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, eps=self.opt.adam_epsilon)
            '''
            w = w - learning_rate * (dw + lambda * w)
            '''
        else:
            '''
            not using different learning rates
            '''
            logger.info("bert learning rate on")

            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.opt.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
                'weight_decay': 0.0}
            ]
            
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.bert_lr, eps=self.opt.adam_epsilon)

        return optimizer
    
    def active_learning(self, opt, iteration, fold1_model_path, fold2_model_path):
        term_count = defaultdict(int)
        total_count = 0
        self.unlabeled_dataloader = DataLoader(dataset=self.unlabeled_dataset, batch_size=opt.batch_size)
        for i_batch, sample_batched in enumerate(self.unlabeled_dataloader):
            terms = sample_batched['term']
            for term in terms:
                term_count[term] += 1
                total_count += 1
        term_ratio = {term: count / total_count for term, count in term_count.items()}
        with torch.no_grad():
            model1 = copy.deepcopy(self.model)  # 创建模型的深拷贝
            model1.load_state_dict(torch.load(fold1_model_path))
            model1.eval()

            model2 = copy.deepcopy(self.model)  # 创建模型的深拷贝
            model2.load_state_dict(torch.load(fold2_model_path))
            model2.eval()
            uncertainties = []
            all_sentence_ids = []
            for i_batch, sample_batched in enumerate(self.unlabeled_dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                sentence_ids = sample_batched['sentence_id'].to(self.opt.device)
                outputs1, _ = model1(inputs)
                outputs2, _ = model2(inputs)
                outputs = (outputs1 + outputs2) / 2
                terms = sample_batched['term']
                all_sentence_ids.extend(sentence_ids.cpu().numpy())
                sentence_outputs = defaultdict(list)
                for sentence_id, output in zip(sentence_ids, outputs):
                    sentence_outputs[sentence_id].append(output)
                
                average_outputs = []
                for sentence_id, output in sentence_outputs.items():
                    average_output = torch.mean(torch.stack(output), dim = 0)
                    average_outputs.append(average_output)
                average_outputs = torch.stack(average_outputs)
                probs = torch.nn.functional.softmax(average_outputs, dim=-1)
                adjustment_factors = [(math.log(1 / term_ratio[term]))**(1/10) for term in terms]
                adjustment_factors = torch.tensor(adjustment_factors).to(self.opt.device)
                uncertainties.extend((adjustment_factors * torch.sum(-probs * torch.log(probs), dim=-1)).cpu().numpy())
    
        num_samples_to_label = 200
        indices_to_label = np.argsort(uncertainties)[-num_samples_to_label:]
        sentence_ids_to_label = [all_sentence_ids[i] for i in indices_to_label]
        self.label_samples(sentence_ids_to_label, opt, iteration)

    def wait_for_labeling(self, opt):
        while True:
            user_input = input("Enter 'completed' to proceed:").lower()
            if user_input == 'completed':
                return True
            else:
                print("Invalid Input")
    def get_latest_labeled_data(self, opt, iteration):
        connection_string = opt.connection_string
        container_name = opt.container_name
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
    
        # 获取容器中所有blobs的列表，每个blob包含它的名称和属性
        blob_list = container_client.list_blobs()
        csv_blobs = [blob for blob in blob_list if blob.name.endswith(".csv")]
        blob_list = sorted(csv_blobs, key=lambda b: b.last_modified, reverse=True)
    
        # 下载最新的blob
        latest_blob = blob_list[0]
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=latest_blob.name)
        with open(f"labeled_samples_{iteration+1}.csv", "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
            
        labeled_samples_df = pd.read_csv(f"labeled_samples_{iteration+1}.csv")
        labeled_samples_df.drop(columns = ['polarity'], inplace = True)
        labeled_samples_df = labeled_samples_df.rename(columns = {'label': 'polarity'})
        print(labeled_samples_df)
        return labeled_samples_df
    def convert_to_serializable(self,sample):
        new_sample = {}
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                new_sample[key] = value.tolist()
            else:
                new_sample[key] = value
        return new_sample
    def label_samples(self, sentence_ids_to_label, opt, iteration):
        samples_to_label = [sample for sample in self.unlabeled_dataset if sample['sentence_id'] in sentence_ids_to_label]
        samples_to_label_df = pd.DataFrame(samples_to_label)
        samples_to_label_df['link'] = samples_to_label_df['text'].apply(lambda x: "https://twitter.com/search?q=" + quote(x[:30])+ "&src=typed_query&f=top")
        samples_to_label_df.to_csv(f'samples_to_label_{iteration+1}.csv', index=False)
        print("Samples Are Available to Download! Upload it to AZURE for Labeling")
        labeling_status = self.wait_for_labeling(opt)
        ################################## Wait for Labeling #############################################
        if labeling_status:
            max_retries = 3  # set a maximum number of retries
            retry_delay = 5  # delay in seconds between retries
            for attempt in range(max_retries):
                try:
                    logger.info("Automatically Getting Data from Azure")
                    labeled_samples_df = self.get_latest_labeled_data(opt, iteration)
                    break  # If successful, break out of the loop
                except Exception as e:  # Capture the specific exception as 'e'
                    logger.error(f"Error fetching labeled data from Azure: {e}")
                    if attempt < max_retries - 1:  # If it's not the last attempt
                        logger.info(f"Retrying {attempt + 2}/{max_retries} in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                    # If we've exhausted retries, ask the user
                        logger.info("Automatically Getting Data from Azure failed, try manually upload")
                        user_input = input("Enter 'Manually Upload':").lower()
                        if user_input == "manually upload":
                            while True:
                                user_input2 = input("Enter 'Uploaded Completed' when ready: ").lower()
                                if user_input2 == 'uploaded completed':
                                    try:
                                        labeled_samples_df = pd.read_csv(f"labeled_samples_{iteration+1}.csv")
                                        break  # Break from the inner while loop once the file is successfully loaded
                                    except Exception as e:
                                        logger.error(f"Error reading uploaded CSV: {e}")
                                        input("Please ensure the file is correctly named and in the appropriate location. Press Enter to retry.")
                                else:
                                    logger.warning("Invalid input. Please try again.")
                            break  # Break from the outer for loop once data is manually uploaded

            # Extract polarity from labeled data
            polarities = labeled_samples_df['polarity'].tolist()

        # Combine polarities with original samples_to_label data
            for i, sample in enumerate(samples_to_label):
                polarity_dict = {'Positive':2, 'Negative':0, 'Neutral':1, 'Irrelevant':3}
                sample['polarity'] = polarity_dict[polarities[i]]
                if sample['polarity'] != 3:
                    self.trainset.append(sample)
            self.unlabeled_dataset = [sample for sample in self.unlabeled_dataset if sample['sentence_id'] not in sentence_ids_to_label]

    
    def closest_to(self, target, numbers = [0,1,2]):
        distances = [abs(num - target) for num in numbers]
        min_distance = min(distances)
    
        # 如果有多个数字与目标数字的距离相同，选择较小的数字
        closest_nums = [num for num, dist in zip(numbers, distances) if dist == min_distance]
        return torch.tensor(min(closest_nums), dtype = torch.float32)

    def _train(self, criterion, opt):
        best_acc = 0
        best_f1 = 0
        fold1_model_path = None
        fold2_model_path = None
        for iteration in range(opt.iteration):
            iteration_best_acc = 0
            iteration_best_f1 = 0
            for fold, (train_index, test_index) in enumerate(kf.split(self.trainset)):
                logger.info(f"Fold {fold + 1}")
                if fold == 0:
                    self.model.load_state_dict(torch.load('initial_model.pth'))
                    logger.info("Initial Model is Loaded for Fold 1")
                else:
                    self.model.load_state_dict(torch.load('initial_model.pth'))
                    logger.info("Initial Model is Loaded for Fold 2")
                optimizer = self.get_bert_optimizer(self.model)
                train_dataset_active = [self.trainset[i] for i in train_index]
                test_dataset_active = [self.trainset[i] for i in test_index]
                self.train_dataloader_active = DataLoader(dataset=train_dataset_active, batch_size=opt.batch_size, shuffle = True)
                self.test_dataloader_active = DataLoader(dataset=test_dataset_active, batch_size =opt.batch_size)          
                global_step = 0
                acc_scores = []
                f1_scores = []
                for epoch in range(self.opt.num_epoch):
                    logger.info('>' * 60)
                    logger.info('epoch: {}'.format(epoch))
                    
                    n_correct, n_total = 0, 0
                    for i_batch, sample_batched in enumerate(self.train_dataloader_active):
                        global_step += 1
                        self.model.train()
                        optimizer.zero_grad()
                        inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                        outputs, penal = self.model(inputs)
                        penal = penal.to(self.opt.device)
                        sentence_ids = sample_batched['sentence_id'].to(self.opt.device)
                        targets = sample_batched['polarity'].to(self.opt.device)
                        
                        sentence_outputs = defaultdict(list)
                        for sentence_id, output in zip(sentence_ids, outputs):
                            sentence_outputs[sentence_id].append(output)
                        average_outputs = []
                        for sentence_id, output in sentence_outputs.items():
                            average_output = torch.mean(torch.stack(output), dim = 0)
                            average_outputs.append(average_output)    
                        average_outputs = torch.stack(average_outputs)
                        
                        #####################################################
                        sentence_targets = defaultdict(list)
                        for sentence_id, target in zip(sentence_ids, targets):
                            sentence_targets[sentence_id].append(target)
                        average_targets = []
                        for sentence_id, target in sentence_targets.items():
                            target_tensor = torch.stack(target).float()
                            average_target = torch.mean(target_tensor)
                            average_target = self.closest_to(target = average_target.item())
                            average_targets.append(average_target)
                        average_targets = torch.tensor(average_targets).long()
                        average_outputs = average_outputs.to(self.opt.device)
                        average_targets = average_targets.to(self.opt.device)
                        #Compute the loss 
                        
                        if self.opt.losstype is not None:
                            loss = criterion(average_outputs, average_targets) + penal
                        else:
                            loss = criterion(average_outputs, average_targets)
        
                        loss.backward()
                        optimizer.step()
        
                        if global_step % self.opt.log_step == 0:
           
                            n_correct += (torch.argmax(average_outputs, -1) == average_targets).sum().item()
                            n_total += len(average_outputs)
          
                            train_acc = n_correct / n_total
                            test_acc, f1 = self._evaluate(self.test_dataloader_active, show_results=False)
                            logger.info('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc, f1))
                            acc_scores.append(test_acc)
                            f1_scores.append(f1)
                avg_acc_scores = sum(acc_scores) / len(acc_scores)
                avg_f1_scores = sum(f1_scores) / len(f1_scores)
                logger.info(f'Fold{fold+1} - Epoch Average test accuracy: {avg_acc_scores}, Epoch Average F1 score: {avg_f1_scores}')
                if fold == 0:
                    fold1_model_path = '../DualGCNbert/state_dict/en/{}_{}_acc_{:.4f}_f1_{:.4f}_iteration{}_fold{}'.format(
                        self.opt.model_name, self.opt.dataset, avg_acc_scores, avg_f1_scores, iteration+1, fold+1)
                    torch.save(self.model.state_dict(), fold1_model_path)
                    logger.info('>> Flod1 model path saved: {}'.format(fold1_model_path))
                else:
                    fold2_model_path = '../DualGCNbert/state_dict/en/{}_{}_acc_{:.4f}_f1_{:.4f}_iteration{}_fold{}'.format(
                        self.opt.model_name, self.opt.dataset, avg_acc_scores, avg_f1_scores, iteration+1, fold+1)
                    torch.save(self.model.state_dict(), fold2_model_path)
                    logger.info('>> Fold2 model path saved: {}'.format(fold2_model_path))
                if avg_f1_scores > iteration_best_f1:
                    iteration_best_f1 = avg_f1_scores
                    iteration_best_acc = avg_acc_scores
                    logger.info(f'Iteration {iteration + 1}: best f1 {iteration_best_f1} best acc {iteration_best_acc}')
            if iteration_best_f1 > best_f1:
                best_f1 = iteration_best_f1
                best_acc = iteration_best_acc
            if iteration < opt.iteration - 1:
                self.active_learning(opt, iteration, fold1_model_path, fold2_model_path)
        return best_acc, best_f1
    def _evaluate(self, test_data, show_results=False):
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        targets_all, outputs_all = None, None
        with torch.no_grad():
            for batch, sample_batched in enumerate(test_data):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                sentence_ids = sample_batched['sentence_id'].to(self.opt.device)
                torch.cuda.empty_cache()
                outputs, penal = self.model(inputs)
                
                sentence_outputs = defaultdict(list)
                for sentence_id, output in zip(sentence_ids, outputs):
                    sentence_outputs[sentence_id].append(output) 
                average_outputs = []
                for sentence_id, output in sentence_outputs.items():
                    average_output = torch.mean(torch.stack(output), dim = 0)
                    average_outputs.append(average_output)
                average_outputs = torch.stack(average_outputs)
                predictions = torch.argmax(average_outputs, -1)
                
                sentence_targets = defaultdict(list)
                for sentence_id, target in zip(sentence_ids, targets):
                    sentence_targets[sentence_id].append(target)
                average_targets = []
                for sentence_id, target in sentence_targets.items():
                    target_tensor = torch.stack(target).float()
                    average_target = torch.mean(target_tensor)
                    average_target = self.closest_to(target = average_target.item())
                    average_targets.append(average_target)
                average_targets = torch.tensor(average_targets).long()
                average_targets = average_targets.to(self.opt.device)
                predictions = predictions.to(self.opt.device)
                n_test_correct += (predictions == average_targets).sum().item()
                n_test_total += len(average_outputs)
                targets_all = torch.cat((targets_all, average_targets), dim=0) if targets_all is not None else average_targets
                outputs_all = torch.cat((outputs_all, average_outputs), dim=0) if outputs_all is not None else average_outputs
                
                '''
                find incorrect 
                incorrect_indices = (predictions != average_targets).nonzero(as_tuple=True)[0]
                incorrect_samples = []
                print(sample_batched)
                for idx in incorrect_indices:
                    incorrect_sentence_id = sentence_ids[idx]
                    incorrect_samples += [sample for sample in sample_batched if sample['sentence_id'] == incorrect_sentence_id]
                '''
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        
        labels = targets_all.data.cpu()
        predic = torch.argmax(outputs_all, -1).cpu()
        if show_results:
            report = metrics.classification_report(labels, predic, digits=4)
            confusion = metrics.confusion_matrix(labels, predic)
            return report, confusion, test_acc, f1

        return test_acc, f1

    def predict_labels(self, dataloader):
        self.model.eval()
        new_data = []
        with torch.no_grad():
            for batch, sample_batched in enumerate(dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                sentence_ids = sample_batched['sentence_id'].to(self.opt.device)
                outputs, _ = self.model(inputs)
                sentence_outputs = defaultdict(list)
                for sentence_id, output in zip(sentence_ids, outputs):
                    sentence_outputs[sentence_id].append(output) 
                    
                average_outputs = []
                sentence_id_order = []
                for sentence_id, output in sentence_outputs.items():
                    average_output = torch.mean(torch.stack(output), dim = 0)
                    average_outputs.append(average_output)
                    sentence_id_order.append(sentence_id)
                average_outputs = torch.stack(average_outputs)
                predictions = torch.argmax(average_outputs, -1)
                
                for sentence_id, prediction in zip(sentence_id_order, predictions):
                # 找到对应的索引
                    idx = sentence_ids.tolist().index(sentence_id)
                    sample_batched['polarity'][idx] = prediction.item()
                filtered_sample = {
                'term': sample_batched['term'],
                'polarity': sample_batched['polarity'].tolist(),
                'sentence_id': sample_batched['sentence_id'].tolist(),
                'text': sample_batched['text']
                }
                new_data.append(filtered_sample)
        return new_data

        
    def run(self, opt):
        criterion = nn.CrossEntropyLoss()
        best_acc, best_f1 = self._train(criterion, opt)
        logger.info(f': best_acc:{best_acc}, best_f1: {best_f1}')
        while True:
            user_input = input("Iteration Completed, check the best model and enter 'predict' to predict unlabeled data:")
            if user_input == 'predict':
                while True:
                    is_used_model = input("Enter 'yes' to use new model to predict when active learning is used. Otherwise, enter 'no'")
                    if is_used_model == 'yes':
                        iteration = input("Enter best model iteration: ")
                        fold = input("Enter best model fold: ")
                    elif is_used_model == 'no':
                        iteration = input("Enter best model iteration: ")
                        fold = input("Enter best model fold: ")
                        best_acc = float(input("Enter best model acc: "))
                        best_f1 = float(input("Enter best model f1: "))
                    best_model_path = "../DualGCNbert/state_dict/en/dualgcnbert_twitter_acc_{:.4f}_f1_{:.4f}_iteration{}_fold{}".format(best_acc, best_f1, iteration, fold)
                    if os.path.exists(best_model_path):
                        self.model.load_state_dict(torch.load(best_model_path))
                        logger.info(f"Best Model Path is loaded {best_model_path}")
                        data_with_labeled = self.predict_labels(self.unlabeled_dataloader)
                        logger.info('Finished Predicting Unlabeled Data')
                        with open("../dataset/labeled/en/en.pkl", 'wb') as f:
                            pickle.dump(data_with_labeled, f)
                        break
                    else:
                        print("invalid model path!")
                break
            else:
                print("invalid input!")
def main():
    model_classes = {
        'dualgcnbert': DualGCNBertClassifier,
    }
    
    dataset_files = {
        'twitter': {
            'initial_en': '../Initial_data/initial_data_en.json',
            'unlabeled_en': '../dataset/unlabeled/en/en.json',
            'initial_cn':'../Initial_data/initial_dataset_cn.json',
            'unlabeled_cn':'../dataset/unlabeled/cn/cn.json',
        }
    }
    
    input_colses = {
        'dualgcnbert': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'adj_matrix', 'src_mask', 'aspect_mask']
    }
    
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    
    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad, 
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax, 
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }
    
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', default=0, type = int, help = 'Num of iteration')
    parser.add_argument('--model_name', default='dualgcnbert', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='twitter', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--log_step', default=20, type=int)
    parser.add_argument('--num_layers', type=int, default=3, help='Num of GCN layers.')
    parser.add_argument('--polarities_dim', default=3, type=int, help='3')
    parser.add_argument('--hidden_dim', type=int, default=768, help='dim')

    parser.add_argument('--gcn_dropout', type=float, default=0.1, help='GCN layer dropout rate.')
    parser.add_argument('--direct', default=False, help='directed graph or undirected graph')
    
    parser.add_argument('--attention_heads', default=1, type=int, help='number of multi-attention heads')
    parser.add_argument('--max_length', default=85, type=int)
    parser.add_argument('--device', default='cuda', type=str, help='cpu, cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight deay if we apply some.")
    parser.add_argument('--cuda', default='0', type=str)
    parser.add_argument('--losstype', default='doubleloss', type=str, help="['doubleloss', 'orthogonalloss', 'differentiatedloss']")
    parser.add_argument('--alpha', default=0.25, type=float)
    parser.add_argument('--beta', default=0.25, type=float)
    parser.add_argument('--parseadj', default=False, action='store_true', help='dependency probability')
    
    # * bert
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--bert_dropout', type=float, default=0.3, help='BERT dropout rate.')
    parser.add_argument('--diff_lr', default=False, action='store_true')
    parser.add_argument('--bert_lr', default=2e-5, type=float)
    
    #Azure Labeling
    parser.add_argument('--connection_string', default = 'your_connection_string', type = str)
    parser.add_argument('--ws_subscrip_id', default = 'your_ws_subscrip_id', type = str)
    parser.add_argument('--container_name', default= 'your_container_name', type = str)
    parser.add_argument('--resource_group', default= 'your_resource_group', type = str)
    parser.add_argument('--ws_name', default = 'your_ws_name', type = str)
    opt = parser.parse_args()
    	
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    # print("choice cuda:{}".format(opt.cuda))
    # os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)
    
    # set random seed
    setup_seed(opt.seed)

    if not os.path.exists('../DualGCNbert/log/en'):
        os.makedirs('../DualGCNbert/log/en', mode=0o777)
    log_file = r'{}_{}_{}.log'.format(opt.model_name, opt.dataset, strftime("%Y_%m_%d_%H_%M_%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % (r'../DualGCNbert/log/en', log_file)))

    ins = Instructor(opt)
    ins.run(opt)

if __name__ == '__main__':
    main()
