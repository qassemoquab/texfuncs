local TexFunRandFlip, parent = torch.class('nn.TexFunRandFlip', 'nn.ExtractInterpolateBDHW')

local help_str = 
[[Horizontally flips an input RGB image flipprob% of the time. Based on ExtractInterpolate.
Doesn't flip at test time.

Usage : m = nxn.TexFunRandFlip(flipprob)
flipprob = 0.5 by default

It only works in BATCH MODE (4D) with RGB inputs :
- with the following input layout : (batch, y, x, RGB).
- RGB are the contiguous dimension.
- a single image must be a (1, y, x, RGB) tensor.

The module doesn't require fixed-size inputs.]]

function TexFunRandFlip:__init(flipprob)
   parent.__init(self)
   self:setFlipProb(flipprob)
end

function TexFunRandFlip:setFlipProb(flipprob)
   self.flipprob=flipprob or 0.5
end

function TexFunRandFlip:updateOutput(input)
   local flip
   
   if self.train then 
      flip=torch.bernoulli(self.flipprob)
   else
      flip=0
   end

   local wInput = input:size(4)
   local hInput = input:size(3)
   
   if flip==0 then
      self.output:resizeAs(input):copy(input)
   else
      if input:type() == 'torch.CudaTensor' then
         local x1 = wInput
         local y1 = 1
         
         local x2 = 1
         local y2 = 1
         
         local x3 = 1
         local y3 = hInput

         local x4 = wInput
         local y4 = hInput

         local targety = hInput
         local targetx = wInput
         
         self:updateOutputCall(input, targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4)
      end
   end

   return self.output

end


