<a name="onmt.GlobalAttention.dok"></a>


## onmt.GlobalAttention ##

 Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.


    H_1 H_2 H_3 ... H_n
     q   q   q       q
      |  |   |       |
       \ |   |      /
           .....
         \   |  /
             a

Constructs a unit mapping:
  $$(H_1 .. H_n, q) => (a)$$
  Where H is of `batch x n x dim` and q is of `batch x dim`.

  The full function is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.



<a class="entityLink" href="https://github.com/opennmt/opennmt/blob/b8ee79ced285a1b7f5720f7e1473e4955a23e9f1/lib/onmt/GlobalAttention.lua#L30">[src]</a>
<a name="onmt.GlobalAttention"></a>


### onmt.GlobalAttention(dim) ###

A nn-style module computing attention.

  Parameters:

  * `dim` - dimension of the context vectors.



#### Undocumented methods ####

<a name="onmt.GlobalAttention:updateOutput"></a>
 * `onmt.GlobalAttention:updateOutput(input)`
<a name="onmt.GlobalAttention:updateGradInput"></a>
 * `onmt.GlobalAttention:updateGradInput(input, gradOutput)`
<a name="onmt.GlobalAttention:accGradParameters"></a>
 * `onmt.GlobalAttention:accGradParameters(input, gradOutput, scale)`
