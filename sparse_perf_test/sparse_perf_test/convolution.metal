//
//  Convolution.metal
//  sparse_perf_test
//
//  Created by Nikita Mishchanka on 5/2/20.
//  Copyright © 2020 Nikita Mishchanka. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void add(const device float2 *in [[ buffer(0) ]],
                device float  *out [[ buffer(1) ]],
                uint id [[ thread_position_in_grid ]]) {
    out[id] = in[id].x + in[id].y;
}

kernel void simple_check_2d(texture2d<float, access::read> inTexture[[texture(0)]],
                                 texture2d<float, access::write> outTexture[[texture(1)]],
                                 ushort3 gid [[thread_position_in_grid]])
{
    if(gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
        return;
    }

    float4 value = float4(0, 0, 0, 0);
    ushort2 inCoord = ushort2(gid.x, gid.y);
    
    for(ushort inComponent = 0; inComponent < 4; inComponent++) {
        float inValue = inTexture.read(inCoord, 0)[inComponent];
        
        value[inComponent] = /*(1 + gid.y) * 10000 + (1 + gid.x * 100) + */(1 + inComponent) * 10 + inValue;
    }
    
    outTexture.write(value, gid.xy, gid.z);
}

kernel void simple_conv_2d(texture2d<float, access::read> inTexture[[texture(0)]],
                           texture2d<float, access::read> convKernel[[texture(1)]],
                           texture2d<float, access::write> outTexture[[texture(2)]],
                           constant int& channelsActive [[buffer(0)]],
                           ushort3 gid [[thread_position_in_grid]]) {
    if(gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
        return;
    }
    
    float4 value = float4(0, 0, 0, 0);
    for(ushort y = 0; y < convKernel.get_height(); y++) {
        for(ushort x = 0; x < convKernel.get_width(); x++) {
            ushort2 inCoord = ushort2(gid.x + x, gid.y + y);
            
            if(inCoord.x >= inTexture.get_width() || inCoord.y >= inTexture.get_height()) {
                continue;
            }
            
            float4 inValue = inTexture.read(inCoord, gid.z);
            float4 kernelValue = convKernel.read(ushort2(x, y), gid.z);
            
            for(ushort inComponent = 0; inComponent < channelsActive; inComponent++) {
                value[inComponent] += inValue[inComponent] * kernelValue[inComponent];
            }
            
        }
    }
    
    outTexture.write(value, gid.xy, gid.z);
}

kernel void sparse_conv_2d(texture2d<float, access::read> inTexture[[texture(0)]],
                           texture2d<float, access::read> convKernel[[texture(1)]],
                           texture1d<float, access::read> weightsIndexes[[texture(2)]],
                           texture2d<float, access::write> outTexture[[texture(3)]],
                           constant int& channelsActive [[buffer(0)]],
                           ushort3 gid [[thread_position_in_grid]]) {
    if(gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() || weightsIndexes.get_width() < 1) {
        return;
    }
    
    float4 value = float4(0, 0, 0, 0);
    uint32_t indexesNum = uint32_t(weightsIndexes.read(ushort(0), 0)[0]);
    
    for(uint32_t weightInd = 2; weightInd < 2 * indexesNum + 2; weightInd += 2) {
        float yInd = weightsIndexes.read(ushort(weightInd), 0)[0];
        float xInd = weightsIndexes.read(ushort(weightInd + 1), 0)[0];
        
        ushort2 inCoord = ushort2(gid.x + xInd, gid.y + yInd);
        float4 inValue = inTexture.read(inCoord, gid.z);
        
        ushort2 kernelCoord = ushort2(xInd, yInd);
        float4 kernelValue = convKernel.read(kernelCoord, gid.z);
        
        for(ushort inComponent = 0; inComponent < channelsActive; inComponent++) {
            value[inComponent] += inValue[inComponent] * kernelValue[inComponent];
        }
    }

    outTexture.write(value, gid.xy, gid.z);
}
