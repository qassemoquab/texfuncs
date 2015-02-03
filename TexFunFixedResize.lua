local TexFunFixedResize, parent = torch.class('nn.TexFunFixedResize', 'nn.ExtractInterpolateBDHW')

local help_str = 
[[Resizes an input RGB image by a fixed scale. Based on ExtractInterpolate.

Usage : m = nxn.TexFunFixedResize(scale)

It only works in BATCH MODE (4D) with RGB inputs :
- with the following input layout : (batch, y, x, RGB).
- RGB are the contiguous dimension.
- a single image must be a (1, y, x, RGB) tensor.

The module doesn't require fixed-size inputs.]]

function TexFunFixedResize:__init(scale)
   parent.__init(self)
   self.scale=scale
end

function TexFunFixedResize:setScale(scale)
   if not scale then
      error('TexFunFixedResize:setScale(scale) or TexFunFixedResize(scale)')
   end
   self.scale=scale or 1
end

function TexFunFixedResize:updateOutput(input)
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




