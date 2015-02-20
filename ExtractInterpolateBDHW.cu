#ifndef assert
#define assert(e)  \
    if (!(e)) { \
        printf("failed assertion `%s'\n", #e); \
        THError("aborting..."); \
    };
#endif

#include "utils.h"

static texture<float, cudaTextureType2DLayered> texRef2;

__global__ void extractInterpolateBDHWKernel(float* outptr, int outstr0, int outstr1, int outstr2, int outstr3, int outx, int outy, float y1, float x1, float y2, float x2, float y3, float x3, float y4, float x4, int inLayers)
{

   const int pixIdxX = blockIdx.x*blockDim.x+threadIdx.x;
   const int pixIdxY = blockIdx.y*blockDim.y+threadIdx.y;

   const float coordx0 = (float)(pixIdxX)/outx;
   const float coordy0 = (float)(pixIdxY)/outy;

   // we put some offset (y_i, x_i) are the input coordinates of the output corners (1 : top-left, 2 : top-right, 3 : bot-right, 4 : bot-left)
   const float upinter = (x1+(coordx0*(x2-x1)));
   const float downinter = (x4+(coordx0*(x3-x4)));
   const float leftinter = (y1+(coordy0*(y4-y1)));
   const float rightinter = (y2+(coordy0*(y3-y2)));

   const float coordx = upinter + coordy0*(downinter - upinter);
   const float coordy = leftinter + coordx0*(rightinter - leftinter);

   int i;

//   if (coordx<1 && coordy<1 && coordx0 <1 && coordy0 <1)
   if(pixIdxX<outx && pixIdxY<outy)
   {
   // read :
      for (i=0; i<inLayers; i++)
      {
          outptr[outstr1*i + outstr2*pixIdxY + outstr3*pixIdxX] = tex2DLayered(texRef2, coordx, coordy, i);
      }
   }
}

static int texfuncs_ExtractInterpolate_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  cudaArray* imgarray = (cudaArray *) lua_touserdata(L, 3);

  int outy = luaT_getfieldcheckint(L, 1, "targety");
  int outx = luaT_getfieldcheckint(L, 1, "targetx");
  int y1int = luaT_getfieldcheckint(L, 1, "y1");
  int y2int = luaT_getfieldcheckint(L, 1, "y2");
  int y3int = luaT_getfieldcheckint(L, 1, "y3");
  int y4int = luaT_getfieldcheckint(L, 1, "y4");
  int x1int = luaT_getfieldcheckint(L, 1, "x1");
  int x2int = luaT_getfieldcheckint(L, 1, "x2");
  int x3int = luaT_getfieldcheckint(L, 1, "x3");
  int x4int = luaT_getfieldcheckint(L, 1, "x4");

  input = THCudaTensor_newContiguous(state, input); // should be contiguous already
  
  int bs       = input->size[0];
  int nPlanes  = input->size[1];
  int ih       = input->size[2];
  int iw       = input->size[3];
//  assert(nPlanes==3);

  THCudaTensor_resize4d(state, output, bs, nPlanes,  outy, outx);
  
  float y1 = ((float)y1int-1)/(float)(ih-1);
  float y2 = ((float)y2int-1)/(float)(ih-1);
  float y3 = ((float)y3int-1)/(float)(ih-1);
  float y4 = ((float)y4int-1)/(float)(ih-1);
  float x1 = ((float)x1int-1)/(float)(iw-1);
  float x2 = ((float)x2int-1)/(float)(iw-1);
  float x3 = ((float)x3int-1)/(float)(iw-1);
  float x4 = ((float)x4int-1)/(float)(iw-1);
  
  
  cudaError_t result;
  cudaError_t err;

  float * outptr=THCudaTensor_data(state, output);
  
    texRef2.addressMode[0]   = cudaAddressModeBorder;
    texRef2.addressMode[1]   = cudaAddressModeBorder;
    texRef2.filterMode       = cudaFilterModeLinear;
    texRef2.normalized       = 1;
	
    cudaBindTextureToArray(texRef2, imgarray);


    int outstr0    = output->stride[0];
    int outstr1    = output->stride[1];
    int outstr2    = output->stride[2];
    int outstr3    = output->stride[3];
    
    dim3 blockstiled((outx+7)/8, (outy+3)/4, bs);
    dim3 threadstiled(8,4);

//    dim3 blocks((outx+31)/32, outy);
//    dim3 threads(32);
    dim3 blocks((outx+7)/8, (outy+7)/8);
    dim3 threads(8,8);


    //printf("%f, %f, %f, %f, %f, %f, %f, %f\n", y1, x1, y2, x2, y3, x3, y4, x4);
    
    extractInterpolateBDHWKernel <<<blocks, threads>>>(outptr, outstr0, outstr1, outstr2, outstr3, outx, outy, y1, x1, y2, x2, y3, x3, y4, x4, bs*nPlanes);

   err = cudaGetLastError();
 
    cudaUnbindTexture(texRef2);

  // check for errors
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ExtractInterpolate.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
 
  // final cut:
  THCudaTensor_free(state, input); 
  //THCudaTensor_free(tmp); 
  //THCudaTensor_select(output, NULL, dimension, 0);

  return 1;
}



static int texfuncs_ExtractInterpolate_initCudaArray(lua_State *L)
{
    THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* imgarray;

    int bs       = input->size[0];
    int nPlanes  = input->size[1];
    int ih       = input->size[2];
    int iw       = input->size[3];
//    assert(nPlanes==3);
 
    cudaExtent ex = make_cudaExtent(iw, ih, bs*nPlanes);

    cudaError_t result;

    result = cudaMalloc3DArray(&imgarray, &channelDesc, ex, cudaArrayLayered);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaMalloc3DArray -  %s\n", cudaGetErrorString(result));
        return 1;
    }  

    lua_pushlightuserdata (L, imgarray);
    return 1;
}


static int texfuncs_ExtractInterpolate_copyIntoArray(lua_State *L)
{
    THCState *state = getCutorchState(L);
    THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
    cudaArray* imgarray = (cudaArray *) lua_touserdata(L, 3);

    int bs       = input->size[0];
    int nPlanes  = input->size[1];
    int ih       = input->size[2];
    int iw       = input->size[3];
//    assert(nPlanes==3);

    cudaError_t result;

    cudaMemcpy3DParms myParms = {0};
    memset(&myParms, 0, sizeof(myParms));
    myParms.srcPtr.pitch = sizeof(float) * iw;
//    myParms.srcPtr.ptr = THCudaTensor_data(state, input);
    myParms.srcPtr.ptr = THCudaTensor_data(state, input);
    myParms.srcPtr.xsize = iw;
    myParms.srcPtr.ysize = ih;

    myParms.srcPos.x = 0;
    myParms.srcPos.y = 0;
    myParms.srcPos.z = 0;
  
    myParms.dstArray = imgarray;

    myParms.dstPos.x = 0;
    myParms.dstPos.y = 0;
    myParms.dstPos.z = 0;

    myParms.extent.width = iw;
    myParms.extent.depth = bs*nPlanes;
    myParms.extent.height = ih;

    myParms.kind = cudaMemcpyDeviceToDevice;

    result = cudaMemcpy3DAsync(&myParms);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy3D - failed to copy 1 - %s\n", cudaGetErrorString(result));
        return 1;
    }
    return 1;
}


static int texfuncs_ExtractInterpolate_destroyArray(lua_State *L)
{
    cudaArray* imgarray = (cudaArray *) lua_touserdata(L, 2);
    cudaError_t result;
    result = cudaFreeArray(imgarray);
    if (result != cudaSuccess) {
        fprintf(stderr, "cudaFreeArray - %s\n", cudaGetErrorString(result));
        return 1;
    }
    return 0;
}





static int texfuncs_ExtractInterpolate_updateGradInput(lua_State *L)
{
  return 1;
}

static const struct luaL_Reg texfuncs_ExtractInterpolate__ [] = {
  {"texfuncs_ExtractInterpolate_updateOutput", texfuncs_ExtractInterpolate_updateOutput},
  {"texfuncs_ExtractInterpolate_initCudaArray", texfuncs_ExtractInterpolate_initCudaArray},
  {"texfuncs_ExtractInterpolate_copyIntoArray", texfuncs_ExtractInterpolate_copyIntoArray},
  {"texfuncs_ExtractInterpolate_destroyArray", texfuncs_ExtractInterpolate_destroyArray},
  {"", },
  {NULL, NULL}
};

static void texfuncs_ExtractInterpolate_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, texfuncs_ExtractInterpolate__, "nn");
  lua_pop(L,1);
}
