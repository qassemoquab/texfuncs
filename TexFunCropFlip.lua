local TexFunCropFlip, parent = torch.class('nn.TexFunCropFlip', 'nn.TexFunCustom')

local function fnTrain(self, input)
   local wInput = input:size(4)
   local hInput = input:size(3)
    
   local targety = self.targety or hInput
   local targetx = self.targetx or wInput
   
   local xcrop = math.random(0,self.cropx)
   local ycrop = math.random(0,self.cropy)

   local x1=1+xcrop
   local y1=1+ycrop
   
   local x2=wInput-self.cropx+xcrop
   local y2=1+ycrop
     
   local x3=wInput-self.cropx+xcrop
   local y3=hInput-self.cropy+ycrop

   local x4=1+xcrop
   local y4=hInput-self.cropy+ycrop
   
   local flip = torch.bernoulli(0.5)
   
   if flip==0 then 
      return targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4
   else
      return targety, targetx, y2, x2, y1, x1, y4, x4, y3, x3
   end
end


local function fnTest(self, input)
   local targety = self.targety or hInput
   local targetx = self.targetx or wInput

   local wInput = input:size(4)
   local hInput = input:size(3)

   local xcrop = math.floor(self.cropx/2)
   local ycrop = math.floor(self.cropy/2)

   local x1=1+xcrop
   local y1=1+ycrop
   
   local x2=wInput-self.cropx+xcrop
   local y2=1+ycrop
     
   local x3=wInput-self.cropx+xcrop
   local y3=hInput-self.cropy+ycrop

   local x4=1+xcrop
   local y4=hInput-self.cropy+ycrop
   
   return targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4
end

function TexFunCropFlip:__init(cropx, cropy, targetx, targety)
    parent.__init(self, fnTrain, fnTest)
    self.cropx = cropx
    self.cropy = cropy or cropx
    self.targetx  = targetx
    self.targety  = targety
end

