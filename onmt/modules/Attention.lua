local Attention, parent = torch.class('onmt.Attention', 'onmt.Network')

--[[ Attention model take a matrix and a query vector. It
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

Attention can be local in which case it applies on a windows centered around p_t, or global.

Attention types are:
  * dot: $$score(h_t,{\overline{h}}_s) = h_t^T{\overline{h}}_s$$
  * general: $$score(h_t,{\overline{h}}_s) = h_t^T W_a {\overline{h}}_s$$
  * concat: $$score(h_t,{\overline{h}}_s) = \nu_a^T tanh(W_a[h_t;{\overline{h}}_s])$$

]]

local options = {
  {
    '-attention', 'global',
    [[Attention model type.]],
    {
      enum = {'none', 'local', 'global'},
      structural = 0
    }
  },
  { '-global_attention', '',
    [[]],
    {
      deprecated = '-attention_model'
    }
  },
  {
    '-attention_model', 'general',
    [[Attention type.]],
    {
      enum = {'general', 'dot', 'concat'},
      structural = 0
    }
  }
}

function Attention.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Attention Model')
end

function Attention:__init(opt, model)
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(opt, options)
  parent.__init(self, model)
end
