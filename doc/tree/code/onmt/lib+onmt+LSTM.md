<a name="onmt.LSTM.dok"></a>


## onmt.LSTM ##


<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/39968aa86f3b4f7a7c93720c38460e10a0f040a4/lib/onmt/LSTM.lua#L17">[src]</a>
<a name="onmt.LSTM"></a>


### onmt.LSTM(num_layers, input_size, hidden_size, dropout) ###

A nn-style module computing one LSTM step.

Computes $$(c_{t-1}, h_{t-1}, x_t) => (c_{t}, h_{t})$$.

Parameters:

  * `num_layers` - Number of LSTM layers.  
  * `input_size` -  Size of input layer x.
  * `hidden_size` -  Size of the hidden layers (cell and hidden).
  * `dropout` - Dropout rate to use.



#### Undocumented methods ####

<a name="onmt.LSTM:updateOutput"></a>
 * `onmt.LSTM:updateOutput(input)`
<a name="onmt.LSTM:updateGradInput"></a>
 * `onmt.LSTM:updateGradInput(input, gradOutput)`
<a name="onmt.LSTM:accGradParameters"></a>
 * `onmt.LSTM:accGradParameters(input, gradOutput, scale)`
