//
//  ViewController.swift
//  sparse_perf_test
//
//  Created by Nikita Mishchanka on 5/2/20.
//  Copyright © 2020 Nikita Mishchanka. All rights reserved.
//

import UIKit
import MetalKit

class ViewController: UIViewController {
    
    @IBOutlet weak var label: UILabel!
    @IBOutlet weak var label2: UILabel!
    @IBOutlet weak var label3: UILabel!
    var device: MTLDevice = MTLCreateSystemDefaultDevice()!

    override func viewDidLoad() {
        super.viewDidLoad()
        
        let debugIndeces = false
        if (debugIndeces)
        {
            let indexTexture = createIndexTexture(height: 3, width: 3, sparse: 0.4)
            
            let buffer1 = convertTextureToBuffer(texture: indexTexture)
            label?.numberOfLines = 0
            label?.lineBreakMode = .byWordWrapping
            label?.text = bufferToString(buffer: buffer1, channels: indexTexture.arrayLength,
                                                            width: 6,
                                                            height: 3)
            label?.sizeToFit()
            label?.font = label.font.withSize(15)
            
            let increasingTex = createIncreasingTexture(channels: 3, height: 3, width: 3)
            let buffer2 = convertTextureToBuffer(texture: increasingTex)
            label2?.numberOfLines = 0
            label2?.lineBreakMode = .byWordWrapping
            label2?.text = bufferToString(buffer: buffer2, channels: increasingTex.arrayLength,
                                                            width: increasingTex.width,
                                                            height: increasingTex.height)
            label2?.sizeToFit()
            label2?.font = label.font.withSize(10)
            
            return
        }
        
        let validateResults = true
        if (validateResults)
        {
            let inputTexture = createNoiseTexture(channels: 10,
                                                       height: 8,
                                                       width: 8)
            let weightTexture = createNoiseTexture(channels: inputTexture.arrayLength,
                                                        height: 3,
                                                        width: 3)
            let indexTexture = createIndexTexture(height: weightTexture.height,
                                                  width: weightTexture.width,
                                                  sparse: 0)
            let outputTexture1 = createEmptyTexture(channels: 1,
                                                   height: inputTexture.height,
                                                   width: inputTexture.width)
            let outputTexture2 = createEmptyTexture(channels: 1,
                                                   height: inputTexture.height,
                                                   width: inputTexture.width)
            
            do {
                let library = device.makeDefaultLibrary()!
                let lib_function = library.makeFunction(name: "simple_conv_2d")!
            
                let pipeline = try! device.makeComputePipelineState(function: lib_function)
                
                let queue = device.makeCommandQueue()!
                let cmds = queue.makeCommandBuffer()!
                let encoder = cmds.makeComputeCommandEncoder()!
                
                encoder.setComputePipelineState(pipeline)
                encoder.setTexture(inputTexture, index: 0)
                encoder.setTexture(weightTexture, index: 1)
                encoder.setTexture(outputTexture1, index: 2)
                
                var activeChannels = 1
                encoder.setBytes(&activeChannels, length: MemoryLayout<Int>.size, index: 0)
                
                let w = pipeline.threadExecutionWidth
                let h = pipeline.maxTotalThreadsPerThreadgroup / w
                let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
                
                let threadgroupsPerGrid = MTLSize(width: (inputTexture.width + w - 1) / w,
                                                  height: (inputTexture.height + h - 1) / h,
                                                  depth: outputTexture1.arrayLength)
                        
                encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                encoder.endEncoding()
                
                cmds.commit()
                cmds.waitUntilCompleted()
            }
            
            do {
                let library = device.makeDefaultLibrary()!
                let lib_function = library.makeFunction(name: "sparse_conv_2d")!

                let pipeline = try! device.makeComputePipelineState(function: lib_function)
                
                let queue = device.makeCommandQueue()!
                let cmds = queue.makeCommandBuffer()!
                let encoder = cmds.makeComputeCommandEncoder()!
                
                encoder.setComputePipelineState(pipeline)
                encoder.setTexture(inputTexture, index: 0)
                encoder.setTexture(weightTexture, index: 1)
                encoder.setTexture(indexTexture, index: 2)
                encoder.setTexture(outputTexture2, index: 3)
                
                var activeChannels = 1
                encoder.setBytes(&activeChannels, length: MemoryLayout<Int>.size, index: 0)
                
                let w = pipeline.threadExecutionWidth
                let h = pipeline.maxTotalThreadsPerThreadgroup / w
                let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
                
                let threadgroupsPerGrid = MTLSize(width: (inputTexture.width + w - 1) / w,
                                                  height: (inputTexture.height + h - 1) / h,
                                                  depth: outputTexture2.arrayLength)
                        
                encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                encoder.endEncoding()
                
                cmds.commit()
                cmds.waitUntilCompleted()
            }
            
            let buffer1 = convertTextureToBuffer(texture: outputTexture1)
            let buffer2 = convertTextureToBuffer(texture: outputTexture2)
            
            if (checkBuffers(buffer1: buffer1, buffer2: buffer2)) {
                label3?.text = "Both convolutions have SAME results!"
            }
            else {
                label3?.text = "Both convolutions have DIFFERENT results!"
            }
            
            return
        }
        
        let useSparse = true
        let printTime = false
        let sparseCoeff = Float(0)
        
        let inputTexture = createIncreasingTexture(channels: 2,
                                                   height: 3,
                                                   width: 3)
        
        let weightTexture = createIncreasingTexture(channels: 2,
                                                    height: 2,
                                                    width: 2)
        let indexTexture: MTLTexture
        if (useSparse) {
            indexTexture = createIndexTexture(height: weightTexture.height,
                                              width: weightTexture.width,
                                              sparse: sparseCoeff)
        }
        else {
            indexTexture = createEmptyTexture(channels: 1,
                                              height: weightTexture.height,
                                                width: weightTexture.width)
        }
        
        let outputTexture = createEmptyTexture(channels: 1,
                                               height: inputTexture.height,
                                               width: inputTexture.width)
        
        let start = DispatchTime.now()
        let library = device.makeDefaultLibrary()!
        let lib_function : MTLFunction
        if (useSparse) {
            lib_function = library.makeFunction(name: "sparse_conv_2d")!
        }
        else {
            lib_function = library.makeFunction(name: "simple_conv_2d")!
        }
        let pipeline = try! device.makeComputePipelineState(function: lib_function)
        
        let queue = device.makeCommandQueue()!
        let cmds = queue.makeCommandBuffer()!
        let encoder = cmds.makeComputeCommandEncoder()!
        
        //----------------------------------------------------------------------
        // encoder
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(inputTexture, index: 0)
        encoder.setTexture(weightTexture, index: 1)
        if (useSparse) {
            encoder.setTexture(indexTexture, index: 2)
            encoder.setTexture(outputTexture, index: 3)
        }
        else {
            encoder.setTexture(outputTexture, index: 2)
        }
        
        var activeChannels = 1
        encoder.setBytes(&activeChannels, length: MemoryLayout<Int>.size, index: 0)
        
        let w = pipeline.threadExecutionWidth
        let h = pipeline.maxTotalThreadsPerThreadgroup / w
        let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
        
        let threadgroupsPerGrid = MTLSize(width: (inputTexture.width + w - 1) / w,
                                          height: (inputTexture.height + h - 1) / h,
                                          depth: outputTexture.arrayLength)
                
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        cmds.commit()
        cmds.waitUntilCompleted()
        let end = DispatchTime.now()
   
        if (printTime) {
            let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds // <<<<< Difference in nano seconds (UInt64)
            let timeInterval = Double(nanoTime) / 1_000_000_000
            label3?.text = "Finished! " + String(timeInterval)
        }
        else {
            let buffer1 = convertTextureToBuffer(texture: inputTexture)
            label?.numberOfLines = 0
            label?.lineBreakMode = .byWordWrapping
            label?.text = bufferToString(buffer: buffer1, channels: inputTexture.arrayLength, width: inputTexture.width, height: inputTexture.height)
            label?.sizeToFit()
            label?.font = label.font.withSize(10)
            
            let buffer2 = convertTextureToBuffer(texture: weightTexture)
            label2?.numberOfLines = 0
            label2?.lineBreakMode = .byWordWrapping
            label2?.text = bufferToString(buffer: buffer2, channels: weightTexture.arrayLength, width: weightTexture.width, height: weightTexture.height)
            label2?.sizeToFit()
            label2?.font = label2.font.withSize(10)
            
            let buffer3 = convertTextureToBuffer(texture: outputTexture)
            label3?.numberOfLines = 0
            label3?.lineBreakMode = .byWordWrapping
            label3?.text = bufferToString(buffer: buffer3, channels: outputTexture.arrayLength, width: outputTexture.width, height: outputTexture.height)
            label3?.sizeToFit()
            label3?.font = label3.font.withSize(10)
        }
    }

    func createIncreasingTexture(channels: Int, height: Int, width: Int) -> MTLTexture {
        let descriptor = MTLTextureDescriptor()
        descriptor.arrayLength = channels //Int(floor(Float(channels + 3) / 4))
        descriptor.height = height
        descriptor.width = width
        descriptor.pixelFormat = MTLPixelFormat.r32Float; //!!!!
        descriptor.textureType = MTLTextureType.type2DArray
        descriptor.usage = [.shaderRead, .shaderWrite]
        
        let texture = device.makeTexture(descriptor: descriptor)!
        
        let region = MTLRegionMake2D(0, 0, width, height)
        let bytesPerRow = MemoryLayout<Float>.size * width;
        let bytesPerImage = bytesPerRow * height
        var counter = Float32(1)
            
        for slice in 0..<descriptor.arrayLength {
            var pixels = [Float](repeating: 0, count: width * height)
//            var pixels = [Float](repeating: 0, count: 4 * width * height)
//            let sliceChannels = min(1, channels - 4 * slice)
            
            for i in 0..<width * height {
                pixels[i] = counter
                counter += 1
                
//                for channel in 0..<sliceChannels {
//                    pixels[i * 4 + channel] = counter
//                    counter += 1
//                }
            }
            
            pixels.withUnsafeBytes{(pointer: UnsafeRawBufferPointer) in
                texture.replace(region: region,
                                mipmapLevel: 0,
                                slice: slice,
                                withBytes: pointer.baseAddress!,
                                bytesPerRow: bytesPerRow,
                                bytesPerImage: bytesPerImage)
            }
            
            pixels.removeAll()
        }
        
        return texture
    }

    func createNoiseTexture(channels: Int, height: Int, width: Int) -> MTLTexture {
        let descriptor = MTLTextureDescriptor()
        descriptor.arrayLength = channels //Int(floor(Float(channels + 3) / 4))
        descriptor.height = height
        descriptor.width = width
        descriptor.pixelFormat = MTLPixelFormat.r32Float; //!!!!
        descriptor.textureType = MTLTextureType.type2DArray
        descriptor.usage = [.shaderRead, .shaderWrite]
        
        let texture = device.makeTexture(descriptor: descriptor)!
        
        let region = MTLRegionMake2D(0, 0, width, height)
        let bytesPerRow = MemoryLayout<Float>.size * width;
        let bytesPerImage = bytesPerRow * height
            
        for slice in 0..<descriptor.arrayLength {
            var pixels = [Float](repeating: 0, count: width * height)
            
            for i in 0..<width * height {
                pixels[i] = Float.random(in: -5...5)
            }
            
//            var pixels = [Float](repeating: 0, count: 4 * width * height)
//            let sliceChannels = min(4, channels - 4 * slice)
//
//            for i in 0..<width * height {
//                for channel in 0..<sliceChannels {
//                    pixels[i * 4 + channel] = Float.random(in: -5...5)
//                }
//            }
            
            pixels.withUnsafeBytes{(pointer: UnsafeRawBufferPointer) in
                texture.replace(region: region,
                                mipmapLevel: 0,
                                slice: slice,
                                withBytes: pointer.baseAddress!,
                                bytesPerRow: bytesPerRow,
                                bytesPerImage: bytesPerImage)
            }
            pixels.removeAll()
        }
        
        return texture
    }
    
    func createEmptyTexture(channels: Int, height: Int, width: Int) -> MTLTexture {
        let descriptor = MTLTextureDescriptor()
        descriptor.arrayLength = channels //Int(floor(Float(channels + 3) / 4))
        descriptor.height = height
        descriptor.width = width
        descriptor.pixelFormat = MTLPixelFormat.r32Float; //!!!!
        descriptor.textureType = MTLTextureType.type2DArray
        descriptor.usage = [.shaderRead, .shaderWrite]
        
        let texture = device.makeTexture(descriptor: descriptor)!
        
        let region = MTLRegionMake2D(0, 0, width, height)
        let bytesPerRow = 1 * MemoryLayout<Float>.size * width; //*4
        let bytesPerImage = bytesPerRow * height
        
        for slice in 0..<descriptor.arrayLength {
            var pixels = [Float](repeating: 0, count: width * height)
            
            for i in 0..<width * height {
                pixels[i] = Float.random(in: -5...5)
            }
            
            pixels.withUnsafeBytes{(pointer: UnsafeRawBufferPointer) in
                texture.replace(region: region,
                                mipmapLevel: 0,
                                slice: slice,
                                withBytes: pointer.baseAddress!,
                                bytesPerRow: bytesPerRow,
                                bytesPerImage: bytesPerImage)
            }
            pixels.removeAll()
        }
        
        return texture
    }
    
    func generateActiveWeights(height: Int, width: Int, sparse: Float) -> [Float] {
        var indexes = [Float](repeating: 0, count: 2 * width * height + 2)
        
        var toNull = Int(Float(height * width) * sparse)
        var toStay = height * width - toNull
        
        indexes[0] = Float(toStay)
        indexes[1] = Float(toNull)
        
        var pointer = Int(1)
        for i in 0..<width * height {
            if (toStay + toNull == 0) {
                break
            }
            
            let random = Float.random(in: 0...1)
            let threshold = Float(toStay) / Float(toStay + toNull)
            
            if (random <= threshold) {
                toStay -= 1
                indexes[2 * pointer] = Float(Int(i) / (width))
                indexes[2 * pointer + 1] = Float(Int(i) % (width))
                pointer += 1
            }
            else {
                toNull -= 1
            }
            
            if (toStay == 0 && pointer < width * height) {
                indexes[2 * pointer] = -1
                indexes[2 * pointer + 1] = -1
                break
            }
        }
        
        return indexes
    }
    
    func createIndexTexture(height: Int, width: Int, sparse: Float) -> MTLTexture {
        let descriptor = MTLTextureDescriptor()
        descriptor.arrayLength = 1
        descriptor.height = 1
        descriptor.width = 2 * height * width + 2 // because x & y
        descriptor.pixelFormat = MTLPixelFormat.r32Float; // maybe rg32Float Only here!
        descriptor.textureType = MTLTextureType.type1D
        descriptor.usage = [.shaderRead, .shaderWrite]
        
        let texture = device.makeTexture(descriptor: descriptor)!
        
        let region = MTLRegionMake1D(0, 2 * width * height + 2)
        let bytesPerRow = MemoryLayout<Float>.size * (2 * width * height + 2); // maybe 2!
        let bytesPerImage = bytesPerRow
        
        for slice in 0..<descriptor.arrayLength {
            var pixels = generateActiveWeights(height: height, width: width, sparse: sparse)
            
            pixels.withUnsafeBytes{(pointer: UnsafeRawBufferPointer) in
                texture.replace(region: region,
                                mipmapLevel: 0,
                                slice: slice,
                                withBytes: pointer.baseAddress!,
                                bytesPerRow: bytesPerRow,
                                bytesPerImage: bytesPerImage)
            }
            pixels.removeAll()
        }
        
        return texture
    }
    
    func convertTextureToBuffer(texture: MTLTexture) -> [Float32] {
        let bytesPerPixel = 4
        let imageByteCount = texture.width * texture.height * texture.arrayLength * texture.depth * bytesPerPixel
        let bytesPerRow = texture.width * bytesPerPixel
        var buffer = [Float32](repeating: 0, count: Int(imageByteCount))

        let region = MTLRegionMake2D(0, 0, texture.width, texture.height)
        
        for slice in 0..<texture.arrayLength {
            texture.getBytes(&buffer[slice * texture.width * texture.height], bytesPerRow: bytesPerRow, bytesPerImage: imageByteCount, from: region, mipmapLevel: 0, slice: slice)
        }
        
        return buffer
    }
    
    func bufferToString(buffer: [Float32], channels: Int, width: Int, height: Int) -> String {
        var long_string = String()
        for y in 0...(height-1) {
            for x in 0...(width-1) {
                if (channels == 1) {
                    long_string += String(format: "%.1f, ", buffer[width * y + x])
                }
                else {
                    long_string += String("[")
                    
                    for slice in 0..<channels {
                        long_string += String(format: "%.1f, ", buffer[slice * width * height + width * y + x])
                    }
                    
                    long_string += String("] ")
                }
            }
            long_string += String("\n")
        }
        
        return long_string
    }
    
    func checkBuffers(buffer1: [Float32], buffer2: [Float32]) -> Bool {
        if (buffer1.count != buffer2.count) {
            return false
        }
        
        for i in 0...(buffer1.count - 1) {
            if (buffer1[i] != buffer2[i]) {
                return false
            }
        }
        
        return true
    }
}
