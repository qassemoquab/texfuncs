texfuncs
==============

Texture-based functions for BDHW layout


Test with : 

```lua

require 'texfuncs'
require 'sys'
require 'image'

ei=nn.TexFunDeformation(0.2,224,224):cuda()
input=torch.CudaTensor(1,3,256,256):uniform()
input:copy(image.scale(image.lena(), 256,256))

sys.tic()
ei:updateOutput(input)
print(sys.toc())

image.display(input)
w=image.display(ei.output)

for i=1,100 do
   sys.tic()
   cutorch.synchronize()
   ei:updateOutput(input)
   cutorch.synchronize()
   print(sys.toc())
   image.display({image = ei.output, win = w})
end


```
