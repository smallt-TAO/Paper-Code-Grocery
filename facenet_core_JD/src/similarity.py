from scipy import spatial
import sys

def cosine_similarity(a, b):
  '''
  cacl cosine similarity
  '''
  return 1 - spatial.distance.cosine(a, b)

def similarity(embedding_file, outputfile, contrast):
  '''
  cacl similarity of embeddings
  '''
  contrast_dict = {}
  with open(contrast, 'r') as lines:
    for line in lines:
      arr = line.strip().split('\t')
      contrast_dict[arr[0].strip()] = arr[1].strip()

  embs_dict = {}
  with open(embedding_file,'r') as lines:
    for line in lines:
      arr = line.strip().split('\t')
      if len(arr) < 2:
        continue
      sku = arr[0]
      emb = [float(x) for x in arr[1].strip().split()]
      if sku in embs_dict:
        embs_dict[sku].append(emb)
      else:
        embs_dict[sku]=[emb]


  print('size of embs:%d' % len(embs_dict))
  outfile = open(outputfile, 'w')
  for sku in embs_dict.keys():
    embs = embs_dict.get(sku)
    if len(embs) != 2:
      outfile.write('sku %s has only one emb\n' % sku)
      continue
    elif len(embs) == 2:
      label = contrast_dict.get(sku, 'unknow')
      sim = cosine_similarity(embs[0], embs[1])
      outfile.write('%s\t%s\t%s\n' % (sku, label, str(sim)))
    else:
      outfile.write('sku %s : emb num:%d' %(sku, len(embs)))

if __name__ == '__main__':
  similarity(sys.argv[1], sys.argv[2], sys.argv[3])
