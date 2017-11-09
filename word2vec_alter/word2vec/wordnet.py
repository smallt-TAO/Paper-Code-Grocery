from nltk.corpus import wordnet

# 获得一个词的所有sense，包括词语的各种变形的sense：

wordnet.synsets('published')

[Synset('print.v.01'),

 Synset('publish.v.02'),

 Synset('publish.v.03'),

 Synset('published.a.01'),

 Synset('promulgated.s.01')]

 

# 得到synset的词性：

>>> related.pos

's'

 

# 得到一个sense的所有lemma：

>>> wordnet.synsets('publish')[0].lemmas

[Lemma('print.v.01.print'), Lemma('print.v.01.publish')]

 

# 得到Lemma出现的次数：

>>> wordnet.synsets('publish')[0].lemmas[1].count()

39

 

# 在wordnet中，名词和动词被组织成了完整的层次式分类体系，因此可以通过计算两个sense在分类树中的距离，这个距离反应了它们的语义相似度：

>>> x = wordnet.synsets('recommended')[-1]

>>> y = wordnet.synsets('suggested')[-1]

>>> x.shortest_path_distance(y)

0

 

# 形容词和副词的相似度计算方法：

形容词和副词没有被组织成分类体系，所以不能用path_distance。

>>> a = wordnet.synsets('beautiful')[0]

>>> b = wordnet.synsets('good')[0]

>>> a.shortest_path_distance(b)

-1

# 形容词和副词最有用的关系是similar to。

>>> a = wordnet.synsets('glorious')[0]

>>> a.similar_tos()

[Synset('incandescent.s.02'),

 Synset('divine.s.06'),

……]
