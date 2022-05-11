import tensorflow as tf
import numpy as np

class SimpleDense(tf.keras.layers.Layer):
    def __init__(self):
        super(SimpleDense, self).__init__()    
        self.trans = np.array([[0.18126767, 0.63919559, 0.73759456, 0.45651186],
        [0.7681438 , 0.29595808, 0.68564719, 0.12727231],
        [0.68696636, 0.05947026, 0.45851196, 0.3310953 ]])
        self.trans = tf.convert_to_tensor(self.trans, dtype=tf.float32)

    def call(self, inputs):
        pts = self.trans @ tf.transpose(inputs, perm=[0, 2, 1])
        
        Z = pts[:, 2, :]
        x = pts[:, 0, :]/Z
        y = pts[:, 1, :]/Z
        x = tf.clip_by_value(x, 0, 37-1)
        y = tf.clip_by_value(y, 0, 120-1)

        depth_img = tf.Variable(tf.zeros((2, 37, 120)), trainable=False)
        for bi in range(2):
            for xi, yi, zi in zip(x[bi], y[bi], Z[bi]):
                if zi > 0:
                    depth_img = depth_img[bi, tf.cast(xi, tf.int32), tf.cast(yi, tf.int32)].assign(zi)
        return depth_img

def main():
    projector = SimpleDense()
    pcs = tf.random.normal((2, 100, 4), 0, 100)
    return projector(pcs)

if __name__ == "__main__":
    print(main())