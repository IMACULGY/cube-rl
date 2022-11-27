import numpy as np
from math import ceil

# map which takes in a color, and outputs an array representing the color
colormap = {
    "y": np.array([1,0,0,0,0,0]),
    "w": np.array([0,1,0,0,0,0]),
    "g": np.array([0,0,1,0,0,0]),
    "b": np.array([0,0,0,1,0,0]),
    "r": np.array([0,0,0,0,1,0]),
    "o": np.array([0,0,0,0,0,1])
}

# we always track the "sticker" which shows up first in the color array
def trackindex(cubelet):
    return np.nonzero(cubelet>0)[0][0]

# map which takes in a string representing the cubelet, and outputs the corresponding indices in the cube string
cubeletlocmap = {
    "UB":(13,88),
    "UR":(35,79),
    "UF":(51,70),
    "UL":(29,61),
    "FL":(104,101),
    "FR":(110,113),
    "BL":(128,95),
    "BR":(122,119),
    "DB":(219,162),
    "DR":(203,153),
    "DF":(181,144),
    "DL":(197,135),
    "UBL":(10,91,58),
    "UBR":(16,85,82),
    "UFR":(54,73,76),
    "UFL":(48,67,64),
    "DBL":(216,165,132),
    "DBR":(222,159,156),
    "DFR":(184,147,150),
    "DFL":(178,141,138)
}

# maps a cubelet color tuple to a cubelet index in the encoded string (edges and corners)
indexmap = {
    (1,0,0,1,0,0):0,
    (1,0,0,0,0,1):1,
    (1,0,1,0,0,0):2,
    (1,0,0,0,1,0):3,
    (0,0,0,1,1,0):4,
    (0,0,0,1,0,1):5,
    (0,0,1,0,0,1):6,
    (0,0,1,0,1,0):7,
    (0,1,1,0,0,0):8,
    (0,1,0,0,0,1):9,
    (0,1,0,1,0,0):10,
    (0,1,0,0,1,0):11,
    (1,0,0,1,1,0):12,
    (1,0,0,1,0,1):13,
    (1,0,1,0,0,1):14,
    (1,0,1,0,1,0):15,
    (0,1,1,0,1,0):16,
    (0,1,1,0,0,1):17,
    (0,1,0,1,0,1):18,
    (0,1,0,1,1,0):19
}


# maps a string index to a location index in the encoded string (edges and corners)
locationmap = {
    # edges
    13:0,
    35:1,
    51:2,
    29:3,
    61:4,
    101:5,
    135:6,
    95:7,
    70:8,
    110:9,
    144:10,
    104:11,
    79:12,
    119:13,
    153:14,
    113:15,
    88:16,
    128:17,
    162:18,
    122:19,
    181:20,
    203:21,
    219:22,
    197:23,
    # corners
    10:0,
    16:1,
    54:2,
    48:3,
    58:4,
    64:5,
    138:6,
    132:7,
    67:8,
    73:9,
    147:10,
    141:11,
    76:12,
    82:13,
    156:14,
    150:15,
    91:16,
    85:17,
    165:18,
    159:19,
    178:20,
    184:21,
    222:22,
    216:23
}

# cubelet keys list
cubeletkeys = list(cubeletlocmap.keys())

# function which takes in a cube and outputs its state encoding
# to represent states, we one-hot encode the location of one sticker per cubelet
# 12 edge cubelets w/ 24 possible locations, 8 corner cubelets w/ 24 possible location
# EX: arr[n][m] = 1 -> the nth cubelet is at location m
def encode(cube):
    cubestr = str(cube)
    arr = np.zeros((20,24))
    for i in range(20):
        s = cubeletkeys[i]
        # get the colors in the cubelet spot
        cubelet = np.array([0,0,0,0,0,0])
        for ind in cubeletlocmap[s]:
            cubelet += (colormap[cubestr[ind]] * ind)
        # use the map to get proper cubelet index n
        print(cubelet)
        n = indexmap[tuple(np.ceil(cubelet / 999.0).astype(int))]
        # use the map to get proper location index m
        m = locationmap[cubelet[trackindex(cubelet)]]
        # update encoding
        arr[n,m] = 1
    return arr

# uncomment to debug
# import pycuber as pc
# cube = pc.Cube()
# print(repr(cube))
# print(encode(cube))
# cube("R U R'")
# print(cube)
# print(repr(cube))
# print(encode(cube))