# Initialization

from Bio import SeqIO
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt

seq = None
gb = None

for record in SeqIO.parse("GCF_000355675.1_ASM35567v1_genomic.fna", "fasta"):
    seq = record.seq

for record in SeqIO.parse("GCF_000355675.1_ASM35567v1_genomic.gbff", 'genbank'):
    gb = record
    
start, end = gb.features[0].location.start, None

ncod = []
cod = []

for i in range(len(gb.features)):
  feature = gb.features[i]
  fseq = feature.extract(seq)

  end = feature.location.start

  if feature.type == 'CDS' and fseq[:3] == 'ATG' and len(fseq) % 3 == 0:
    cod.append(fseq.__str__())

    if len(seq[start:end]) != 0:
      ncod.append(seq[start:end].__str__())

    start = feature.location.end

## 2. Probabilities calculation
# 2.1 Initial Probabilities

def seq_probs(seq):
  return np.array([
      seq.count('T'), seq.count('C'), 
      seq.count('A'), seq.count('G')
    ]) / len(seq)

def cod_probs(seq):
  res = []
  for i in range(3):
    res.append(seq_probs(seq[i::3]))
  return np.array(res)

def make_table1(cod_seqs, ncod_seqs):
  table1 = pd.DataFrame(np.vstack((cod_probs(''.join(cod_seqs)), seq_probs(''.join(ncod_seqs)))).T, 
                       index=['T', 'C', 'A', 'G'], 
                       columns=[f'pos{i}' for i in range(1, 4)] + ['nc'])
  return table1

# 2.2 Transition probabilities

def cod_dprobs(seqs):
  dcounts = dict(
    zip([1, 2, 3], 
        [dict(zip([''.join(pair) for pair in product('TCAG', repeat=2)],
                  [0] * 16)) for i in range(3)])
    )
  for seq in seqs:
    for i in range(1, len(seq)):
      dcounts[i % 3 + 1][seq[i-1:i+1]] += 1
  return get_probs(dcounts)

def ncod_dprobs(seqs):
    dcounts =dict(zip([''.join(pair) for pair in product('TCAG', repeat=2)],
                    [0] * 16))
    for seq in seqs:
      for i in range(1, len(seq)):
        dcounts[seq[i-1:i+1]] += 1
        
    return get_probs({0: dcounts})[0]

def get_probs(dcounts):
  for pos in dcounts:
    nuc_groups = dict(zip('TCAG', [0] * 4))
    for dup in dcounts[pos]:
      nuc_groups[dup[0]] += dcounts[pos][dup]
    for dup in dcounts[pos]:
      dcounts[pos][dup] /= nuc_groups[dup[0]]
  return dcounts

def make_table2(cod_seqs, ncod_seqs):
  table2 = pd.DataFrame(cod_dprobs(cod_seqs))
  table2[4] = pd.Series(ncod_dprobs(ncod_seqs))
  table2.rename(columns=dict(zip(np.arange(1, 5), 
                                 [f'pos{i}' for i in range(1, 4)] + ['nc'])),
                inplace=True)
  table2.index = [prob_notation(idx) for idx in table2.index.values]
  return table2

def prob_notation(st):
  return st[1] + '|' + st[0]

t1 = make_table1(cod, ncod)
print('\nВ данной таблице указаны буквы, кодирующие аминокислоты, и вероятности их появления на первой, второй и третьей позиции кодона (триплета). \nСтолбец "nc" отражает частоту появления этих букв в некодирующих участках генома\n')
print(t1)

print('\nДанная таблица составлена по аналогии с первой, за исключением учёта условной вероятности нахождения буквы после буквы, кодирующей одну из других аминокислот\n')
t2 = make_table2(cod, ncod)
print(t2)

## 3. Predictions
# 3.1 Calculating predictions

def cod_proba(seq, t1, t2, frame = 1):
  
  if len(seq) == 0:
    print('No sequence')
    return None
  
  if frame not in [1, 2, 3]:
    print('No such frame')
    return None

  prev_elem = seq[0]
  start_pos = None
  
  if frame == 1:
    log_prob = np.log(t1['pos1'][seq[0]])
    prev_pos = 1
    for index, elem in enumerate(seq[1:]):
      pair = '{}|{}'.format(elem, prev_elem)
      if prev_pos == 1:
        log_prob += np.log(t2['pos2'][pair])
        prev_pos = 2
      elif prev_pos == 2:
        log_prob += np.log(t2['pos3'][pair])
        prev_pos = 3
      else:
        log_prob += np.log(t2['pos1'][pair])
        prev_pos = 1
      prev_elem = elem
    prob = np.exp(log_prob)
    return prob

  elif frame == 2:
    log_prob = np.log(t1['pos3'][seq[0]])
    prev_pos = 3
    for index, elem in enumerate(seq[1:]):
      pair = '{}|{}'.format(elem, prev_elem)
      if prev_pos == 1:
        log_prob += np.log(t2['pos2'][pair])
        prev_pos = 2
      elif prev_pos == 2:
        log_prob += np.log(t2['pos3'][pair])
        prev_pos = 3
      else:
        log_prob += np.log(t2['pos1'][pair])
        prev_pos = 1
      prev_elem = elem
    prob = np.exp(log_prob)
    return prob 
    
  elif frame == 3:
    log_prob = np.log(t1['pos2'][seq[0]])
    prev_pos = 2
    for index, elem in enumerate(seq[1:]):
      pair = '{}|{}'.format(elem, prev_elem)
      if prev_pos == 1:
        log_prob += np.log(t2['pos2'][pair])
        prev_pos = 2
      elif prev_pos == 2:
        log_prob += np.log(t2['pos3'][pair])
        prev_pos = 3
      else:
        log_prob += np.log(t2['pos1'][pair])
        prev_pos = 1
      prev_elem = elem
    prob = np.exp(log_prob)
    return prob 

def ncod_proba(seq, t1, t2):
  if len(seq) == 0:
    print('No sequence')
    return None
  
  log_prob_nc = np.log(t1['nc'][seq[0]])
  prev_elem = seq[0]
  for index, elem in enumerate(seq[1:]):
    pair = '{}|{}'.format(elem, prev_elem)
    log_prob_nc += np.log(t2['nc'][pair ])
    prev_elem = elem
  prob_nc = np.exp(log_prob_nc)
  return prob_nc

def get_cod_probs(seq, t1, t2):
  prob1 = cod_proba(seq, t1, t2, frame=1)
  prob2 = cod_proba(seq, t1, t2, frame=2)
  prob3 = cod_proba(seq, t1, t2, frame=3)
  probnc = ncod_proba(seq, t1, t2)

  res1 = (0.25 * prob1)/(0.25 * prob1 + 0.25 * prob2 + 0.25 * prob3 + 0.25 * probnc)
  res2 = (0.25 * prob2)/(0.25 * prob1 + 0.25 * prob2 + 0.25 * prob3 + 0.25 * probnc)
  res3 = (0.25 * prob3)/(0.25 * prob1 + 0.25 * prob2 + 0.25 * prob3 + 0.25 * probnc)
  resnc = (0.25 * probnc)/(0.25 * prob1 + 0.25 * prob2 + 0.25 * prob3 + 0.25 * probnc)
  return [res1, res2, res3, resnc]

## 3.2 Vizualizing

sequence = seq[2000:5996]
codon_list = ["ATG", "TAA", "TAG", "TGA"]

found_codon_positions = []

n = len(sequence)
k = 0
while k < n-2:
    possible_codon = sequence[k:k+3]
    if possible_codon in codon_list:
        found_codon_positions.append(k)
    k += 1
    
def plot_graph(data, start, end, step):
  fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))

  x = range(start, end, step)

  ax1.plot(x, data[:,0])
  ax2.plot(x, data[:,1])
  ax3.plot(x, data[:,2])

  ax1.set_title('Codon position 1')
  ax2.set_title('Codon position 2')
  ax3.set_title('Codon position 3')

  plt.plot()

pos_probs = list()
seq1 = seq[2000:5996]
start = 0
end = 3000
step = 12
window = 96

for i in range(start, end, step):
  pos_probs.append(get_cod_probs(seq1[i:i+window], t1, t2))
pos_probs = np.array(pos_probs)

plot_graph(pos_probs, start, end, step)

plt.show()