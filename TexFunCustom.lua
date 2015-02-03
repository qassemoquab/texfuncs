local TexFunCustom, parent = torch.class('nn.TexFunCustom', 'nn.ExtractInterpolate')

local help_str = 
[[Custom version of ExtractInterpolate, where user should specify :
- fn(input) should return for training images : targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4
- fntest(input) should return for test images : targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4

Usage : m = nxn.TexFunCustom(fn, fntest)

It only works in BATCH MODE (4D) with RGB inputs :
- with the following input layout : (batch, y, x, RGB).
- RGB are the contiguous dimension.
- a single image must be a (1, y, x, RGB) tensor.

The module doesn't require fixed-size inputs.]]


function TexFunCustom:__init(fn, fntest)
   parent.__init(self)
   self:setFunc(fn, fntest)
end

function TexFunCustom:setFunc(fn, fntest)
   if fn then
      self.fn=fn
      self.fnTest=fntest or self.exampleFnTest
   else
      print('TexFunCustom:setFunc(fn[, fntest]) or TexFunCustom(fn[, fntest]) where : targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4 = fn(input)')
      print('using default fn (it does scale jitter [0.8, 1.2], 50% flip, 10% deformation jittering)')
      self.fn=self.exampleFn
      self.fnTest=self.exampleFnTest
   end
end

function TexFunCustom:updateOutput(input)
   local targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4 
   if self.train then 
      targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4 = self.fn(input)
   else
      targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4 = self.fnTest(input)
   end
   self:updateOutputCall(input, targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4)
   return self.output
end

function TexFunCustom.exampleFn(input)
   local scale = torch.uniform(0.8,1.2)
   local targety = scale * input:size(2)
   local targetx = scale * input:size(3)
   
   local xcrop = math.floor(input:size(3) * 0.1)
   local ycrop = math.floor(input:size(2) * 0.1)

   local x1 = 1 + math.random(0, xcrop)
   local y1 = 1 + math.random(0, ycrop)
   
   local x2 = input:size(3) - math.random(0, xcrop)
   local y2 = 1 + math.random(0, ycrop)
   
   local x3 = input:size(3) - math.random(0, xcrop)
   local y3 = input:size(2) - math.random(0, ycrop)
   
   local x4 = 1 + math.random(0, xcrop)
   local y4 = input:size(2) - math.random(0, ycrop)
   
   local flip = torch.bernoulli(0.5)
   
   if flip==0 then 
      return targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4
   else
      return targety, targetx, y2, x2, y1, x1, y4, x4, y3, x3
   end
end


function TexFunCustom.exampleFnTest(input)
   local targety = input:size(2)
   local targetx = input:size(3)
   local x1 = 1
   local y1 = 1
   local x2 = input:size(3)
   local y2 = 1
   local x3 = input:size(3)
   local y3 = input:size(2)
   local x4 = 1
   local y4 = input:size(2)
   return targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4
end


