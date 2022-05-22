import os, random, pickle
# # Get all PDB IDs
# files = list(set([a.split('-')[0] for a in os.listdir('../data/dataset')]))
# # Shuffle
# random.shuffle(files)
# # Split 70/30
# training = files[:7 * len(files) // 10]
# validation = files[7 * len(files) // 10:]
# # Save the IDs to be consistent
# pickle.dump(training, open('../results/training.pkl', 'wb'))
# pickle.dump(validation, open('../results/validation.pkl', 'wb'))

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdmolfiles import MolFromMol2File
from rdkit.Chem.rdmolops import GetAdjacencyMatrix, RemoveHs

# Define the atom types we are interested in for the protein
atomDict = {'C': np.int8(1), 'O': np.int8(2), 'N': np.int8(3), 'S': np.int8(4), 'P': np.int8(5), 'H': np.int8(6), 'X': np.int8(7)}
rAtomDict = {v: k for (k,v) in atomDict.items()}
rAtomDict[np.int8(0)] = None

# Define atom types and bond orders we are interested in for the ligand
# We encode aromatics as a separate bond type (1.5: np.int8(4))
bondType = {0.0: np.int8(0), 1.0: np.int8(1), 2.0: np.int8(2), 3.0: np.int8(3), 1.5: np.int8(4)}
bondMap = np.vectorize(lambda x: bondType[x])
# The None entry in ligAtom is necessary to ensure correct one-hot encoding
ligAtom = {None: 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'X': 6}
ligAtomMap = np.vectorize(lambda x: ligAtom[x.GetSymbol()] if x.GetSymbol() in ligAtom else 6)

# Create the images to be used
def createProteinImage(PDBID, imageSize=10, resolution=2):
    # Load the protein by parsing the .pdb file
    with open('data/dataset/{}.rec.pdb'.format(PDBID)) as f:
        # The 77th character is the atom type, while the 30th through 53 characters are the 3D coordinates in angstroms
        protein = [(i[77], np.array([float(k) for k in (i[30:38],i[38:46], i[46:54])])) \
                        for i in f.read().split('\n') if i[:4] == 'ATOM']
    # Use rdkit to load the ligand as well
    ligand = MolFromMol2File('../data/dataset/{}.lig.mol2'.format(PDBID), sanitize=False)
    # rdkit is unable to process some percentage of structures
    if ligand is None:
        return None
    # remove explicit hydrogens
    ligand = RemoveHs(ligand, sanitize=False)
    # Get the centroid of the ligand
    centroid = np.mean(ligand.GetConformer().GetPositions(), axis=0)
    # Calculate the lower bound on atom coordinates that will end up in the protein image
    lower = centroid - np.repeat(imageSize // 2 * resolution, 3)
    # Translate the protein so that the lower bound corresponds to grid index [0,0,0]
    protein = [(i[0], i[-1]-lower) for i in protein]
    # Convert 3D coordinates to grid indices
    protein = [(i, (j // resolution).astype(np.int8)) for (i,j) in protein]
    # Filter atoms which are within the imageSize x imageSize x imageSize box
    protein = [i for i in protein if np.all(i[1] >= 0) and np.all(i[1] < imageSize)]
    protImage = np.zeros((imageSize, imageSize, imageSize, len(atomDict)+1), dtype=bool)
    protImage[:, :, :, 0] = True

    for (i,j) in protein:
        protImage[j[0], j[1], j[2], :] = [k == atomDict[i] for k in range(len(atomDict)+1)]
        
    # Get the ligand adjacency matrix with bond orders
    adj = GetAdjacencyMatrix(ligand, useBO=True)
    
    # Now remove atoms with lowest bond order until a maximum of 36 atoms remain
    toRemove = []
    while adj.shape[0] > 36:
        # Get the bond orders of each atom
        sums = np.sum(adj, axis=0)
        # Find a minimum bond order atom
        i = np.argmin(sums)
        # Remove that atom from the adjacency matrix
        adj = np.concatenate((adj[:i,:], adj[i+1:,:]), axis=0)
        adj = np.concatenate((adj[:,:i], adj[:,i+1:]), axis=1)
        # Keep track of which atoms have been removed from the adjacency matrix
        toRemove.append(i)
    
    # Standardize the adjacency matrices to a 36x36 matrix
    if adj.shape[0] < 36:
        result = np.zeros((36, 36))
        result[:len(adj), :len(adj)] = adj
        adj = result
        
    # Convert bond orders to categories
    bonds = bondMap(adj)
    # One-hot encode adjacency matrix
    bonds = np.array([[[j == t for t in range(len(bondType))] for j in i] for i in bonds])
    # Process the atom types
    atomList = list(ligAtomMap(ligand.GetAtoms()))
    # Remove atoms in the same order they were removed from the adjacency matrix
    for i in toRemove:
        del atomList[i]
    # Standardize to 36 atoms
    atoms = np.zeros(36)
    atoms[:len(atomList)] = atomList
    # One-hot encode atom types
    atoms = np.array([[i == t for t in range(len(ligAtom))] for i in atoms])
    return protImage, (bonds, atoms)

# p, (b, a) = createProteinImage('1a4k')
# print('Protein pocket input shape: ', p.shape)
# print('Number of protein atoms: ', np.sum(p[..., 1:]))
# print('Ligand adjacency matrix shape: ', b.shape)
# print('Number of bonds: ', np.sum(b[..., 1:]))
# print('Atom type vector shape: ', a.shape)
# print('Number of ligand atoms: ', np.sum(a[:, 1:]))

import pickle
(trainingData, medusa, training) = pickle.load(open('data/tutorialData.pkl', 'rb'))

from typeguard import typechecked
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, Conv3D, \
    LayerNormalization, Add, Flatten, Concatenate, Reshape, MaxPooling3D, Layer, \
        AveragePooling3D, Conv3DTranspose, Softmax, Embedding, BatchNormalization
from tensorflow.keras import Model

initializer = tf.keras.initializers.VarianceScaling(scale=0.1)

class SpectralNormalization(tf.keras.layers.Wrapper):
    """Performs spectral normalization on weights.
    This wrapper controls the Lipschitz constant of the layer by
    constraining its spectral norm, which can stabilize the training of GANs.
    See [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957).
    ```python
    net = SpectralNormalization(
        tf.keras.layers.Conv2D(2, 2, activation="relu"),
        input_shape=(32, 32, 3))(x)
    net = SpectralNormalization(
        tf.keras.layers.Conv2D(16, 5, activation="relu"))(net)
    net = SpectralNormalization(
        tf.keras.layers.Dense(120, activation="relu"))(net)
    net = SpectralNormalization(
        tf.keras.layers.Dense(n_classes))(net)
    ```
    Arguments:
      layer: A `tf.keras.layers.Layer` instance that
        has either `kernel` or `embeddings` attribute.
      power_iterations: `int`, the number of iterations during normalization.
    Raises:
      AssertionError: If not initialized with a `Layer` instance.
      ValueError: If initialized with negative `power_iterations`.
      AttributeError: If `layer` does not has `kernel` or `embeddings` attribute.
    """

    @typechecked
    def __init__(self, layer: tf.keras.layers, power_iterations: int = 1, **kwargs):
        super().__init__(layer, **kwargs)
        if power_iterations <= 0:
            raise ValueError(
                "`power_iterations` should be greater than zero, got "
                "`power_iterations={}`".format(power_iterations)
            )
        self.power_iterations = power_iterations
        self._initialized = False

    def build(self, input_shape):
        """Build `Layer`"""
        super().build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

        if hasattr(self.layer, "kernel"):
            self.w = self.layer.kernel
        elif hasattr(self.layer, "embeddings"):
            self.w = self.layer.embeddings
        else:
            raise AttributeError(
                "{} object has no attribute 'kernel' nor "
                "'embeddings'".format(type(self.layer).__name__)
            )

        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name="sn_u",
            dtype=self.w.dtype,
        )

    def call(self, inputs, training=None):
        """Call `Layer`"""
        if training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            self.normalize_weights()

        output = self.layer(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    @tf.function
    def normalize_weights(self):
        """Generate spectral normalized weights.
        This method will update the value of `self.w` with the
        spectral normalized value, so that the layer is ready for `call()`.
        """

        w = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u = self.u

        with tf.name_scope("spectral_normalize"):
            for _ in range(self.power_iterations):
                v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True))
                u = tf.math.l2_normalize(tf.matmul(v, w))

            sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)

            self.w.assign(self.w / sigma)
            self.u.assign(u)

    def get_config(self):
        config = {"power_iterations": self.power_iterations}
        base_config = super().get_config()
        return {**base_config, **config}

# Training/evaluation batch size
BATCHSIZE = 16
# Number of ligand atoms
N = 36

def energy():
    # Protein encoder
    start = Input((10,10,10,8))
    pooled = Flatten()(start)
    for i in [1024]:
        for j in range(10):
            intermediate = pooled
            dropout = Dropout(0.2)(intermediate)
            dense = SpectralNormalization(Dense(i, kernel_initializer=initializer))(dropout)
            act = LeakyReLU()(dense)
            intermediate = act
            if j == 0:
                pooled = intermediate
            else:
                pooled += intermediate

    # Ligand encoder
    bonds = Input((N, N, 5))
    atoms = Input((N, 7))
    s = Concatenate()([Flatten()(bonds), Flatten()(atoms)])
    for i in [1024]:
        for j in range(10):
            intermediate = s
            dropout = Dropout(0.2)(intermediate)
            dense = SpectralNormalization(Dense(i, kernel_initializer=initializer))(dropout)
            act = LeakyReLU()(dense)
            intermediate = act
            # Skip connection
            if j == 0:
                s = intermediate
            else:
                s += intermediate

    # Affinity predictor
    s = Concatenate()([pooled, s])
    for i in [1024]:
        for j in range(10):
            intermediate = s
            dropout = Dropout(0.2)(intermediate)
            dense = SpectralNormalization(Dense(i, kernel_initializer=initializer))(dropout)
            act = LeakyReLU()(dense)
            intermediate = act
            # Skip connection
            if j == 0:
                s = intermediate
            else:
                s += intermediate

    # 13 energies, 7 summary statistics
    bd = SpectralNormalization(Dense(1, kernel_initializer=initializer))(s)
    s = SpectralNormalization(Dense(13*7, kernel_initializer=initializer))(s)

    return Model(inputs=[start, bonds, atoms], outputs=[s, bd])

class Energy(tf.keras.Model):
    def __init__(self):
        super(Energy, self).__init__()
        self.energy = energy()
        self.optimizer = tf.keras.optimizers.Adam(0.000001)

model = Energy()
import time, random, sys
@tf.function
def train_step(receptor, bonds, atoms, trueEnergy, bd, model):
    with tf.GradientTape() as t:
        # Predict MedusaDock statistics (energy) and pK (bdPred)
        energy, bdPred = model.energy([receptor, bonds, atoms])
        
        # L^2 cost for both
        cost = tf.reduce_mean(tf.square(energy - trueEnergy), axis=-1)
        costbd = tf.square(bd-bdPred)
        
        # Evaluate gradient
        grad = t.gradient([cost, costbd], model.energy.trainable_variables)
        
        # Gradient clipping
        grad, _ = tf.clip_by_global_norm(grad, 0.5)
        model.optimizer.apply_gradients(zip(grad, model.energy.trainable_variables))
        
        # Return RMSE per batch
        return tf.sqrt(tf.reduce_mean(cost[..., 0])), tf.sqrt(tf.reduce_mean(costbd))

def print_and_save(s, fname):
    print(s)
    with open(fname,"a") as f:
        f.write(s + "\n")

def main():
    import pickle
    EPOCH = 10000
    with open('data/tutorialData.pkl', 'rb') as f:
        (trainingData, medusa, training) = pickle.load(f)
    #trainingData = {i: (X, Y, Z, bd) for (i, (X, Y, Z, bd)) in trainingData.items()}
    print("Start training...")
    start_epoch = 0
    if start_epoch:
        model.load_weights(f'data/finalNoTrans/model-epoch{start_epoch}')
        print(f"Load model weights from epoch {start_epoch}")
    for epoch in range(start_epoch, EPOCH):
        start = time.time()

        # Shuffle data
        random.shuffle(training)
        batches = len(training) // BATCHSIZE
        
        # Keep track of costs
        c = []
        cbd = []
        curr_log = f"epoch {epoch+1}\t"
        for batch in range(batches):
            
            sIds = training[BATCHSIZE * batch:min(BATCHSIZE * (batch + 1), len(training))]

            receptor, bonds, atoms, bd = zip(*[trainingData[sId] for sId in sIds])

            receptor = np.concatenate(receptor).astype(np.float32)
            bonds = np.stack(bonds).astype(np.float32)
            atoms = np.stack(atoms).astype(np.float32)

            # MedusaDock
            trueEnergy = np.concatenate([medusa[sId] for sId in sIds], axis=0).astype(np.float32)
            
            bd = np.array(bd).astype(np.float32)[:, np.newaxis]

            cost, costbd = train_step(receptor, bonds, atoms, trueEnergy, bd, model)
            cost = cost.numpy()
            costbd = costbd.numpy()
            
            # Keep track of costs
            c.append(cost)
            cbd.append(costbd)
            
            # print(f"Batch {epoch} \tloss: {cost:.4f} \tl bd loss: {costbd:.4f}")
        

        print('Time for epoch {} is {} sec'.format(epoch+1, time.time()-start))
        curr_log += f"loss:{np.mean(c):.4f}\t bdloss:{np.mean(cbd):.4f}"
        print_and_save(curr_log, "pdb-cla.txt")
        sys.stdout.flush()
        
        if (epoch+1) % 2000 == 0:
            print(f"Exporting trained model at epoch {epoch+1}...")
            model.save_weights(f'../data/finalNoTrans/model-epoch{epoch+1}')
    print("Done training!")

if __name__ == "__main__":
    main()
