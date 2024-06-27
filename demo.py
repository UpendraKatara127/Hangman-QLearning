import torch
import torch.nn as nn
from torch.optim import Adam
import random
import torch.nn.functional as F
from tqdm import tqdm
import multiprocess as mp
import numpy as np
from six.moves import range
import six
from collections import namedtuple
import string
from argparse import ArgumentParser
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm
from warnings import filterwarnings
import threading
filterwarnings('ignore')
from collections import defaultdict
from collections import Counter

losses = []
set_letters = set(string.ascii_lowercase)
letters = list(set_letters)
letters.sort()
letters.append('-')
letter_dict = {l : i+1 for i, l in enumerate(letters)}
letter_dict['-'] = 27

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def read_words_from_file(file_path):
    with open(file_path, 'r') as file:
        words = file.read().splitlines()
    return words


words = read_words_from_file('./words_250000_train.txt')
train_val_split_idx = int(len(words) * 0.8)
train_words = words[:train_val_split_idx]
train_words = sorted(train_words, key=lambda x: random.random())
train_words = [word for word in words if len(word) > 9]
print("train_words : {}",len(train_words))
val_words = words[train_val_split_idx:]
val_words = sorted(val_words, key=lambda x: random.random())

print(f'Training with {len(train_words)} words')
print(f'Validation with {len(val_words)} words')

def create_char_counters(words):
    char_counters = defaultdict(Counter)
    for word in words:
        total_counter = Counter(word.lower())
        for char in word.lower():
            tmp = total_counter.copy()
            tmp[char] -= 1
            char_counters[char] = tmp
    return char_counters

result = create_char_counters(words)

def return_merged(input_word):
    input_word = input_word.lower()
    merged_dict = result[input_word[0]]
    for char in input_word[1:]:
        merged_dict.update(result[input_word[1]])
    return merged_dict

def return_feat(input_word):
    input_word = "".join([i.lower() for i in input_word if i!='-'])
    if len(input_word)<1:
        return torch.tensor([1/26 for _ in range(26)], dtype=torch.float32)
    merged_dict = return_merged(input_word)
    out_feat = np.array([merged_dict[i] for i in letters[:-1]])
    out_feat = out_feat/sum(out_feat)
    out_feat = torch.tensor([out_feat], dtype=torch.float32)
    return out_feat

class Network(nn.Module) :
    def __init__(self, curr_word = None, prev_guessed = [], maxlen = 10):

        super().__init__()
        self.curr_word = curr_word
        self.prev_guessed = prev_guessed
        self.lstm1 = torch.nn.LSTM(300, 30, bidirectional=True,dropout=0.2)
        self.lstm2 = torch.nn.LSTM(30, 40, dropout=0.2)
        self.dense1 = torch.nn.Linear(27, 40)
        self.dense2 = torch.nn.Linear(80, 26)
        self.dense3 = torch.nn.Linear(26*2, 26)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()
        self.embedding = nn.Embedding(27, 300, max_norm=True)
        self.full_model = torch.nn.Sequential(self.lstm2, self.dense1, self.activation1, self.dense2, self.activation2)
    
    def return_pred(self, state):
        # print("FORWARDPASS")
        state_str, curr_seqs, prev_guessed = state[0], state[1], state[2]
        #curr_seqs = [i for i in curr_seqs if i!=26]
        curr_seqs = self.embedding.weight.clone()[curr_seqs][:]
        state_embed = self.lstm1(curr_seqs)
        state_embed = state_embed[-1][0]
        state_embed = self.lstm2(state_embed)
        state_embed = state_embed[-1][0]
        state_embed = nn.Tanh()(state_embed)        
        prev_guessed = torch.tensor(prev_guessed, dtype=torch.float32)
        prev_embed = self.dense1(prev_guessed)
        prev_embed = self.activation1(prev_embed)
        
        combined_embed = self.dense2(torch.concat([state_embed, prev_embed], axis=1))
        combined_embed = self.activation1(combined_embed)
        
        n_chars = return_feat(state_str) 
        if len(n_chars.shape)<2:
            n_chars = n_chars.unsqueeze(0)
        output = self.dense3(torch.concat([combined_embed, n_chars], axis=1))
        output = self.activation2(output)
        
        return output

    def __call__(self, state_str, state_seq, guessed) :
        return self.return_pred([state_str,state_seq,guessed]) #self.full_model.predict([state,guessed]).flatten()

    def fit(self, *args, **kwargs) :
        pass #self.full_model.fit(*args, **kwargs)

    def train_on_batch(self, reward) :        
        pass #self.full_model.train_on_batch(*args, **kwargs)

    def summary(self) :
        pass #self.full_model.summary()

    def save(self, *args, **kwargs) :
        self.full_model.save(*args, **kwargs)

    def load_weights(self, *args, **kwargs) :
        self.full_model.load_weights(*args, **kwargs)
        self.compile()

    def compile(self, optimizer= None) :
        print(self.full_model.train())
        return self.full_model.compile()

class Agent(object) :
    def __init__(self, model, policy = 'greedy') :
        self.reset_guessed()
        if policy not in ['stochastic', 'greedy'] :
            raise ValueError('Policy can only be stochastic or greedy')
        self._policy = 'greedy'
        self.policy = property(self.get_policy, self.set_policy)
        self.reset_guessed()
        self.is_training = True
        self.model = model

    @staticmethod
    def guessed_mat(state, guessed) :
        mat = np.zeros([1,27])
        for i, l in enumerate(letters) :
            mat[0,i] = 1 if (l in guessed or l in state) else 0
        return mat

    def get_guessed_mat(self, state) :
        return self.guessed_mat(state, self.guessed)

    def reset_guessed(self) :
        self.guessed = []

    def get_probs(self, state) :
        raise NotImplementedError()

    def get_policy(self) :
        return self._policy

    def set_policy(self, policy) :
        if policy not in ['stochastic', 'greedy'] :
            raise ValueError('Policy can only be stochastic or greedy')
        self._policy = policy

    def select_action(self,state) :
        # print("INSELECTACTION")
        probs = self.get_probs(state)
        if self._policy == 'greedy' :
            i = 1
            sorted_probs = probs.argsort()
            sorted_probs = torch.tensor(sorted_probs[0], dtype=torch.int) 
            while letters[sorted_probs[-i]] in self.guessed :
                i+= 1
            idx_act = sorted_probs[-i]
        elif self._policy == 'stochastic' :
            idx_act = np.random.choice(np.arange(probs.shape[0]), p = probs)
        guess = letters[idx_act]
        if guess not in self.guessed :
            self.guessed.append(guess)
        return guess, probs

    def eval(self) :
        self.is_training = False
        self.set_policy('greedy')

    def train(self) :
        self.is_training = True

class NNAgent(Agent) :
    def __init__(self, model, maxlen=29, policy='greedy') :
        super().__init__(model, policy)
        self.episode_memory = []
        self.states_history = []
        self.maxlen = maxlen

    def train_model(self, reward):
        # print("INORIGINALTRAIN")
        inp_1, inp_2, obj = zip(*self.states_history)
        inp_1 = np.vstack(list(inp_1)).astype(float)
        inp_2 = np.vstack(list(inp_2)).astype(float)
        obj = np.vstack(list(obj)).astype(float)
        loss = self.model.train_on_batch([inp_1,inp_2], obj)
        self.states_history = []
        return loss

    def get_probs(self, state) :
        # print("INGETPROBS")
        state = self.preprocess_input(state)
        probs = self.model(*state)
        probs = probs / probs.sum()
        return probs

    def finalize_episode(self, answer) :
        # print("INFINALISE")
        inp_1, inp_2 = zip(*self.episode_memory)
        inp_1 = np.vstack(list(inp_1)).astype(float)      #stack the game state matrix
        inp_2 = np.vstack(list(inp_2)).astype(float)      #stack the one hot-encoded guessed matrix
        obj = 1.0 - inp_2                                 #compute the unused letters one-hot encoded
        len_ep = len(self.episode_memory)                 #length of episode
        correct_mask = np.array([[1 if l in answer else 0 for l in letters]]) # get mask from correct answer
        # print("Correct mask")
        # print(correct_mask)
        correct_mask = np.repeat(correct_mask, len_ep, axis = 0).astype(float)
        # print("Repeated")
        # print(correct_mask.shape)
        obj = correct_mask * obj  #the correct action is choosing the letters that are both unused AND exist in the word
        obj /= obj.sum(axis = 1).reshape(-1,1) #normalize so it sums to one
        # print("OBJ")
        # print(obj,obj.shape)
        self.states_history.append((inp_1, inp_2,obj))
        self.episode_memory = []
        self.reset_guessed()
        return obj

    def preprocess_input(self, state) :
        # print("INPREPROCESS")
        new_input = []
        new_input_store = []
        for l in state :
            val_idx = letter_dict[l]
            curr_seq = np.zeros(27)
            curr_seq[val_idx-1] = 1
            new_input_store.append(letter_dict[l]-1)
            new_input.append(curr_seq)
        state_store = pad_sequences([new_input_store], maxlen = self.maxlen)
        if self.is_training :
            self.episode_memory.append((state_store,self.get_guessed_mat(state)))
        return state, new_input_store, self.get_guessed_mat(state)

class Hangman(object) :
    def __init__(self ,
                 word_src,
                 max_lives = 6 ,
                 win_reward = 1000, #30
                 correct_reward = 200, #1
                 repeated_guessing_penalty = -100, #-100
                 lose_reward = -2500, #0
                 false_reward = -500, #0
                 verbose = False) :
        if type(word_src) == list :
            self.words = word_src
        else :
            with open(word_src, 'r') as f :
                self.words = f.read().splitlines()
        self.max_lives = max_lives
        self.win_reward = win_reward
        self.correct_reward = correct_reward
        self.lose_reward = lose_reward
        self.false_reward = false_reward
        self.verbose = verbose
        self.repeated_guessing_penalty = repeated_guessing_penalty

    def pick_random(self) :
        self.guess_word = np.random.choice(self.words)

    def reset(self) :
        self.curr_live = self.max_lives
        self.pick_random()
        self.guessing_board = ['-' for i in range(len(self.guess_word))]
        self.correct_guess = 0
        self.guessed = []
        self.done = False
        if self.verbose :
            print('Game Starting')
            print('Current live :', self.curr_live)
        return self.show_gameboard()


    def show_gameboard(self) :
        board = ''.join(self.guessing_board)
        if self.verbose:
            print(board)
            print()
        return board

    def step(self, letter) :
        if not(letter.isalpha()) :
            raise TypeError('Can only accept alphabet')
        letter = letter.lower()

        if letter not in self.guessed :
            self.guessed.append(letter)
        else :
            if self.verbose :
                print('Word used already')
            return self.show_gameboard(), self.repeated_guessing_penalty, self.done, {}


        if letter in self.guess_word :
            for i in range(len(self.guess_word)) :
                if letter == self.guess_word[i] :
                    self.guessing_board[i] = letter
                    self.correct_guess += 1
            if self.correct_guess == len(self.guess_word) :
                self.done = True
                if self.verbose :
                    print('You Win')
                    print('Word is', self.guess_word)
                return self.guess_word, self.win_reward, self.done, {'ans' : self.guess_word}
            else :
                return self.show_gameboard(), self.correct_reward, self.done, {}
        else :
            self.curr_live -= 1
            if self.curr_live == 0 :
                self.done = True
                if self.verbose :
                    print('You Lose')
                    print('Word is', self.guess_word)
                return self.show_gameboard(), self.lose_reward, self.done, {'ans' : self.guess_word}
            else :
                if self.verbose :
                    print('Current lives :', self.curr_live)
                return self.show_gameboard(), self.false_reward, self.done, {}

def process_episode_warm(_):
        optimizer = torch.optim.Adam(player.model.parameters(), lr=0.000005, weight_decay = 1e-5)
        avg_correct = 0
        wins_avg = 0
        state = env.reset()
        done = False
        correct_count = 0
        optimizer.zero_grad()
        while not done:
            guessed = player.get_guessed_mat(state)
            guess, probs = player.select_action(state)
            state, reward, done, ans = env.step(guess)
            print(state)
            correct_mask_answer = np.array([1 if l in env.guess_word else 0 for l in alphabet])
            correct_mask_guessed = np.array([1 if l in guessed else 0 for l in alphabet])
            obj = (1 - correct_mask_guessed) * correct_mask_answer
            obj = obj[:-1]
            if reward > 0:
                correct_count += 1.0
            if reward == env.win_reward:
                wins_avg += 1.0
            target_tensor = torch.from_numpy(obj)#.to(torch_device)
            target = target_tensor.unsqueeze(0).float()
            loss = loss_function(probs, target)
            loss = 0.8*loss + 0.2*(-reward)
            loss.backward()
        losses.append(loss)
        # optimizer.step()
        player.finalize_episode(ans['ans'])
        avg_correct += correct_count
        print("Episode Done")

        return avg_correct, wins_avg

def initialize_worker(train_words, max_lives, Network, NNAgent, Hangman, loss_func, letters, device,lock):
    global env, player, maxlen, loss_function, alphabet, network_class, agent_class, env_class, torch_device, global_lock

    # Assign passed parameters to global variables
    torch_device = device
    loss_function = loss_func
    alphabet = letters
    network_class = Network
    agent_class = NNAgent
    env_class = Hangman
    max_lives = max_lives
    global_lock = lock
    len_list = list(map(len, train_words))
    maxlen = max(len_list)

    # Instantiate environment and player
    global_lock.acquire()
    policy_net = network_class(maxlen=maxlen).to(torch_device)
    player = agent_class(policy_net)
    env = env_class(train_words, max_lives)
    global_lock.release()

def process_episode(_):
        optimizer = torch.optim.Adam(player.model.parameters(), lr=0.00006, weight_decay = 1e-5)
        avg_correct = 0
        wins_avg = 0
        state = env.reset()
        done = False
        correct_count = 0
        optimizer.zero_grad()
        while not done:
            guessed = player.get_guessed_mat(state)
            guess, probs = player.select_action(state)
            state, reward, done, ans = env.step(guess)
            print(state)
            correct_mask_answer = np.array([1 if l in env.guess_word else 0 for l in alphabet])
            correct_mask_guessed = np.array([1 if l in guessed else 0 for l in alphabet])
            obj = (1 - correct_mask_guessed) * correct_mask_answer
            obj = obj[:-1]
            if reward > 0:
                correct_count += 1.0
            if reward == env.win_reward:
                wins_avg += 1.0
            target_tensor = torch.from_numpy(obj)#.to(torch_device)
            target = target_tensor.unsqueeze(0).float()
            loss = loss_function(probs, target)
            loss.backward()
        losses.append(loss)
        optimizer.step()
        player.finalize_episode(ans['ans'])
        avg_correct += correct_count
        torch.save(player.model.embedding.state_dict(), './demo2_embedding')
        torch.save(player.model.lstm1.state_dict(), './demo2_lstm1')
        torch.save(player.model.lstm2.state_dict(), './demo2_lstm2')
        torch.save(player.model.dense1.state_dict(), './demo2_dense1')
        torch.save(player.model.dense2.state_dict(), './demo2_dense2')
        torch.save(player.model.dense3.state_dict(),'./demo2_dense3') 
        # views = (episode_set + 1,avg_correct/(update_episode*view_episode), view_episode, wins_avg/(update_episode*view_episode))

        return avg_correct, wins_avg

total_dict = create_char_counters(words)
words = [word.lower() for word in words]
letter_counts = Counter(''.join(words))
total_letters = sum(letter_counts.values())
letter_frequencies = {letter: count / total_letters for letter, count in letter_counts.items()}
letter_weights = {letter: 1.0 / freq for letter, freq in letter_frequencies.items()}
mean_weight = sum(letter_weights.values()) / len(letter_weights)
letter_weights = {letter: weight / mean_weight for letter, weight in letter_weights.items()}
alphabet = string.ascii_lowercase
class_weights = np.ones(len(alphabet))
for letter, weight in letter_weights.items():
    if (letter in alphabet):  
        index = alphabet.index(letter)
        class_weights[index] = weight
    else:
        pass

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

if __name__=="__main__":
    # Partition words into training and validation sets with validation set being 20% of the total size of the dataset.
    train_val_split_idx = int(len(words) * 0.8)
    train_words = words[:train_val_split_idx]
    train_words = sorted(train_words, key=lambda x: random.random())
    val_words = words[train_val_split_idx:]
    val_words = sorted(val_words, key=lambda x: random.random())

    print(f'Training with {len(train_words)} words')
    print(f'Validation with {len(val_words)} words')
    len_list = list(map(len, train_words))
    maxlen = max(len_list)
    print('Max length of words is', maxlen)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_processes = 1
    episode_sets = range(100)
    update_episode = 5
    policy_net = Network(maxlen=maxlen)  # Define or import your Network class
    player = NNAgent(policy_net)  # Define or import your NNAgent class
    env = Hangman(train_words, 8)  # Define or import your Hangman class
    loss_func = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
    lock = threading.Lock()

    ######################No backprop train###########################
    max_lives = 8
    n_trials = 20000
    ctx = mp.get_context('spawn')
    
    # Initialize workers with shared parameters
    with ctx.Pool(
        processes=num_processes,
        initializer=initialize_worker,
        initargs=(train_words, max_lives, Network, NNAgent, Hangman, loss_func, letters, device,lock)
    ) as pool:
        progbar = tqdm(pool.imap(process_episode_warm, range(n_trials)), total=n_trials)
        results = list(progbar)

    avg_correct_total = 0
    wins_avg_total = 0

    print(results)
    for avg_correct, wins_avg in results:
        avg_correct_total += avg_correct
        wins_avg_total += wins_avg

    print(f'Total Average Correct Count: {avg_correct_total / len(results)}, Total Wins Avg: {wins_avg_total / len(results)}, Total Wins : {wins_avg_total}')

    ######################Final Run####################################
    # Create a context with the 'spawn' start method
    max_lives = 6
    n_trials = 100000
    ctx = mp.get_context('spawn')
    
    # Initialize workers with shared parameters
    with ctx.Pool(
        processes=num_processes,
        initializer=initialize_worker,
        initargs=(train_words, max_lives, Network, NNAgent, Hangman, loss_func, letters, device,lock)
    ) as pool:
        progbar = tqdm(pool.imap(process_episode, range(n_trials)), total=n_trials)
        results = list(progbar)

    avg_correct_total = 0
    wins_avg_total = 0

    print(results)
    for avg_correct, wins_avg in results:
        avg_correct_total += avg_correct
        wins_avg_total += wins_avg
 