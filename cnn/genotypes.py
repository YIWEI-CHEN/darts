from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

# The following architecture is from cifar10_rll_uniform_07_gpu-20190828-112325/log.txt at Epoch 40
RLL_UNIFORM_07_EPOCH40 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 3), ('avg_pool_3x3', 0), ('dil_conv_5x5', 4), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

RLL001_UNIFORM_02 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 4), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
RLL001_UNIFORM_04 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
RLL001_UNIFORM_06 = Genotype(normal=[('max_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
CCE_UNIFORM_06 = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('dil_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_3x3', 4), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('skip_connect', 4)], reduce_concat=range(2, 6))
CCE_UNIFORM_04 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 2), ('skip_connect', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
CCE_UNIFORM_02 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 3)], reduce_concat=range(2, 6))
CCE_UNIFORM_06_SEED_2019 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

RLL001_UNIFORM_06_2019 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('avg_pool_3x3', 0), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))
RLL001_UNIFORM_06_1989 = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 0), ('skip_connect', 2), ('dil_conv_3x3', 1), ('sep_conv_5x5', 3), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 4), ('dil_conv_3x3', 0)], reduce_concat=range(2, 6))
CCE_UNIFORM_06_SEED_1989 = Genotype(normal=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3), ('skip_connect', 1), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

CCE_HIER_06_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 3), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
RLL001_HIER_06_1 = Genotype(normal=[('max_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

# Handcraft architecture
CHECK_REDUCE_IMPACT = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('dil_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_3x3', 4), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
CHECK_NORMAL_IMPACT = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('skip_connect', 4)], reduce_concat=range(2, 6))

# Arch for ranking exp., they are from cifar10_rll0.01_uniform_0.6_gpu-20190906-220956/log.txt
# Epoch 40, valid_acc 38.5
N1 = RLL001_UNIFORM_06

# Epoch 9, valid_acc 27.34
N2 = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))

# Epoch 1, valid_acc 16.9
N3 = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 3), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))


DARTS = DARTS_V2

