require "torch"
require "cunn"
require "libtexfuncs"

local texfuncs = {}

texfuncs.ExtractInterpolateBDHW = require 'texfuncs.ExtractInterpolateBDHW'
texfuncs.TexFunRandResize = require 'texfuncs.TexFunRandResize'
texfuncs.TexFunRandFlip = require 'texfuncs.TexFunRandFlip'
texfuncs.TexFunFixedResize = require 'texfuncs.TexFunFixedResize'
texfuncs.TexFunCustom = require 'texfuncs.TexFunCustom'
texfuncs.TexFunCropJitter = require 'texfuncs.TexFunCropJitter'
texfuncs.TexFunCropFlip = require 'texfuncs.TexFunCropFlip'
texfuncs.TexFunDeformation = require 'texfuncs.TexFunDeformation'

return texfuncs
