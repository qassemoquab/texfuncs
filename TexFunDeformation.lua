local TexFunDeformation, parent = torch.class('nn.TexFunDeformation', 'nn.TexFunCustom')

local function fnTrain(self, input)
   local wInput = input:size(4)
   local hInput = input:size(3)
    
   local targety = self.targety or hInput
   local targetx = self.targetx or wInput
   
   local xcrop = math.floor(hInput * self.strength/2)
   local ycrop = math.floor(wInput * self.strength/2)

   local x1 = 1 + math.random(-xcrop, xcrop)
   local y1 = 1 + math.random(-ycrop, ycrop)
   
   local x2 = wInput - math.random(-xcrop, xcrop)
   local y2 = 1 + math.random(-ycrop, ycrop)
   
   local x3 = wInput - math.random(-xcrop, xcrop)
   local y3 = hInput - math.random(-ycrop, ycrop)
   
   local x4 = 1 + math.random(-xcrop, xcrop)
   local y4 = hInput - math.random(-ycrop, ycrop)
   
   local flip = torch.bernoulli(0.5)
   
   if flip==0 then 
      return targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4
   else
      return targety, targetx, y2, x2, y1, x1, y4, x4, y3, x3
   end
end


local function fnTest(self, input)
   local wInput = input:size(4)
   local hInput = input:size(3)

   local targety = self.targety or hInput
   local targetx = self.targetx or wInput
 
   local x1 = 1
   local y1 = 1
   local x2 = wInput
   local y2 = 1
   local x3 = wInput
   local y3 = hInput
   local x4 = 1
   local y4 = hInput

   return targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4
end

function TexFunDeformation:__init(strength, targetx, targety)
    parent.__init(self, fnTrain, fnTest)
    self.strength = strength or 0.2
    self.targetx  = targetx
    self.targety  = targety
end

