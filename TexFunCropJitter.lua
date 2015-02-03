local TexFunCropJitter, parent = torch.class('nn.TexFunCropJitter', 'nn.ExtractInterpolateBDHW')

local help_str = 
[[Crops (cropx, cropy) pixels from the borders of an image. Based on ExtractInterpolate.
At training time, crops a random patch.
At test time, crops center patch.

Usage : m = nxn.TexFunCropJitter(cropx, cropy)

It only works in BATCH MODE (4D) with RGB inputs :
- with the following input layout : (batch, y, x, RGB).
- RGB are the contiguous dimension.
- a single image must be a (1, y, x, RGB) tensor.

The module doesn't require fixed-size inputs.]]


function TexFunCropJitter:__init(cropx, cropy)
   parent.__init(self)
   self:setCrops(cropx, cropy)
	self.gradInput=torch.Tensor(1)
end

function TexFunCropJitter:setCrops(cropx, cropy)
   if not cropx then
      error('TexFunFixedResize:setCrops(cropx[, cropy]) or TexFunFixedResize(cropx[, cropy])')
   end
   self.cropx=cropx
   self.cropy=cropy or cropx
end

function TexFunCropJitter:updateOutput(input)
   local xcrop, ycrop
   if self.train then
      xcrop = math.random(0,self.cropx)
      ycrop = math.random(0,self.cropy)
   else
      xcrop = math.floor(self.cropx/2)
      ycrop = math.floor(self.cropy/2)
   end

   local wInput = input:size(4)
   local hInput = input:size(3)


   if input:type() == 'torch.CudaTensor' then
      local x1=xcrop
      local y1=ycrop
      
      local x2=wInput-self.cropx+xcrop
      local y2=ycrop
      
      local x3=wInput-self.cropx+xcrop
      local y3=hInput-self.cropy+ycrop

      local x4=xcrop
      local y4=hInput-self.cropy+ycrop

      local targety = hInput-self.cropy
      local targetx = wInput-self.cropx
      
      self:updateOutputCall(input, targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4)
   end   
   return self.output
end




function TexFunCropJitter:updateGradInput(input, gradOutput)
   return self.gradInput
end
