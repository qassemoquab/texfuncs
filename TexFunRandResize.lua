local TexFunRandResize, parent = torch.class('nn.TexFunRandResize', 'nn.ExtractInterpolateBDHW')

local help_str = 
[[Resizes an image by a random factor uniformly sampled in (scale1,scale2). Based on ExtractInterpolate.
At test time, resizes by factor testscale.

Usage : m = nxn.TexFunRandResize(scale1, scale2, testscale)

It only works in BATCH MODE (4D) with RGB inputs :
- with the following input layout : (batch, y, x, RGB).
- RGB are the contiguous dimension.
- a single image must be a (1, y, x, RGB) tensor.

The module doesn't require fixed-size inputs.]]

function TexFunRandResize:__init(scale1, scale2, testscale)
   parent.__init(self)
   self:setScales(scale1, scale2, testscale)
end

function TexFunRandResize:setScales(scale1, scale2, testscale)
   if not scale1 and scale2 and testscale then
      error('TexFunRandResize(scale1, scale2, testscale) or TexFunRandResize:setScales(scale1, scale2, testscale)')
   end
   self.scale1=scale1
   self.scale2=scale2
   self.testscale=testscale
end

function TexFunRandResize:updateOutput(input)
   
   if self.train then 
      self.scale=torch.uniform(self.scale1,self.scale2)
   else
      self.scale=self.testscale 
   end

   local wInput = input:size(4)
   local hInput = input:size(3)
   
   if input:type() == 'torch.CudaTensor' then
      local x1=1
      local y1=1
      
      local x2=wInput
      local y2=1
      
      local x3=wInput
      local y3=hInput

      local x4=1
      local y4=hInput

      local targety = hInput*self.scale
      local targetx = wInput*self.scale
      
      self:updateOutputCall(input, targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4)
   end   
   return self.output

end


