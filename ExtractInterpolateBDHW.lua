local ExtractInterpolate, parent = torch.class('nn.ExtractInterpolateBDHW', 'nn.Module')
-- this is a general class for extraction stuff
-- jitter / resize modules should inherit this class

local help_str = 
[[This is the generic extract/interpolate module. It uses textures on GPU.
It takes an arbitrarily-shaped quadrilateral (by choosing 4 corners) of an RGB image and bilinearly interpolates it to the target size.

Usage : m = nxn.ExtractInterpolate()
m:updateOutputCall(input, targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4)

Where : targety, targetx : size of output
y1,x1 = input pixel corresponding to top left corner of output
y2,x2 = input pixel corresponding to top right corner of output
y3,x3 = input pixel corresponding to bottom right corner of output
y4,x4 = input pixel corresponding to bottom left corner of output

It only works in BATCH MODE (4D) with RGB inputs :
- with the following input layout : (batch, RGB, y, x).
- RGB are the contiguous dimension.
- a single image must be a (1, RGB, y, x) tensor.

The module doesn't require fixed-size inputs.]]

function ExtractInterpolate:__init()
    parent.__init(self)
    self.train = true

    self.gradInput=torch.Tensor(1)
    self.cudaArray=nil
    self.inputSize=nil
end

function ExtractInterpolate:initArray(input)
    self.inputSize=input:size()
    self.cudaArray=input.nn.texfuncs_ExtractInterpolate_initCudaArray(self, input)
end

function ExtractInterpolate:destroyArray()
    self.inputSize=nil
    torch.CudaTensor.nn.texfuncs_ExtractInterpolate_destroyArray(self, self.cudaArray)
end

function ExtractInterpolate:copyIntoArray(input)
    input.nn.texfuncs_ExtractInterpolate_copyIntoArray(self, input, self.cudaArray)
end

function ExtractInterpolate:updateOutput(input)
   return self.output
end


function ExtractInterpolate:updateGradInput(input, gradOutput)
   return 
end

function ExtractInterpolate:updateOutputCall(input, targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4)
   if self.inputSize==nil or (not self.cudaArray) then
      self:initArray(input)
   else
      local _inputSize=input:size()
      for i=1,#self.inputSize do
         if self.inputSize[i] ~= _inputSize[i] then
            self:destroyArray()
            self:initArray(input)
            break
         end
      end
   end
 
   self:copyIntoArray(input)

   -- targety, targetx : size of output
   self.targety=targety
   self.targetx=targetx

   -- y1,x1 = input pixel corresponding to top left corner of output
   -- y2,x2 = input pixel corresponding to top right corner of output
   -- y3,x3 = input pixel corresponding to bottom right corner of output
   -- y4,x4 = input pixel corresponding to bottom left corner of output
   self.y1=y1
   self.y2=y2
   self.y3=y3
   self.y4=y4
   self.x1=x1
   self.x2=x2
   self.x3=x3
   self.x4=x4

   --self.output:resize(input:size(1), self.targety, self.targetx, input:size(4))
   input.nn.texfuncs_ExtractInterpolate_updateOutput(self, input, self.cudaArray)
   
   collectgarbage()

   return 
end

function ExtractInterpolate:updateGradInput(input, gradOutput)
   return self.gradInput
end







