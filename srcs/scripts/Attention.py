from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, MaxPooling1D, Flatten, Dense, MultiHeadAttention

"""
    Definition of our Self-Attention Model
"""
class AttentionModel(Model):
    def __init__(self, n_classes : int):
        super(AttentionModel, self).__init__()
        """ Add LSTM to generate states and sequences"""
        self.lstm0 = LSTM(128, return_sequences=True, name='lstm0')
        self.lstm1 = LSTM(128, return_sequences=True, name='lstm0')

        """Add the attention block"""
        self.attention_block = MultiHeadAttention(num_heads=2, key_dim=128)
        self.pooling = MaxPooling1D(4)

        """Reduce to only one dimension"""
        self.flatten = Flatten()

        """Generate a classification layer with n_classes """
        self.classification = Dense(n_classes, activation='softmax', name='fc0')

    def call(self, x_input):
        x = self.lstm0(x_input)
        x = self.lstm1(x)
        """ If the attention block is performed with the same query 
            value and key the operation is self-attention"""
        x = self.attention_block(query=x, key=x, value=x)
        x = self.pooling(x)
        x = self.flatten(x)
        return self.classification(x)
