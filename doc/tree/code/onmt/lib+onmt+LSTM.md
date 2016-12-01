<a name="onmt.LSTM.dok"></a>


## onmt.LSTM ##


Implementation of a single stacked-LSTM step as
an nn unit.

      h^L_{t-1} --- h^L_t
      c^L_{t-1} --- c^L_t
                 |


                 .
                 |
             [dropout]
                 |
      h^1_{t-1} --- h^1_t
      c^1_{t-1} --- c^1_t
                 |
                 |
                x_t

Computes $$(c_{t-1}, h_{t-1}, x_t) => (c_{t}, h_{t})$$.



<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/b8ee79ced285a1b7f5720f7e1473e4955a23e9f1/lib/onmt/LSTM.lua#L35">[src]</a>
<a name="onmt.LSTM"></a>


### onmt.LSTM(num_layers, input_size, hidden_size, dropout) ###


Parameters:

  * `num_layers` - Number of LSTM layers, $$L$$.
  * `input_size` - Size of input layer,  $$|x|$$.
  * `hidden_size` - Size of the hidden layers (cell and hidden, $$c, h$$).
  * `dropout` - Dropout rate to use.



#### Undocumented methods ####

<a name="onmt.LSTM:updateOutput"></a>
 * `onmt.LSTM:updateOutput(input)`
<a name="onmt.LSTM:updateGradInput"></a>
 * `onmt.LSTM:updateGradInput(input, gradOutput)`
<a name="onmt.LSTM:accGradParameters"></a>
 * `onmt.LSTM:accGradParameters(input, gradOutput, scale)`
