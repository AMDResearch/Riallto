// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// Declare this MLIR module. A block encapsulates all 
// AIE tiles, buffers, and communication in an AI Engine design
module @denoise_aie2 {

 	AIE.device(ipu) {
        // declare kernel external kernel function 
        func.func private @rgba2grayLine(%in: memref<7680xui8>, %out: memref<1920xui8>, %tileWidth: i32) -> ()
        func.func private @median1DLine(%in: memref<1920xui8>, %out: memref<1920xui8>, %lineWidth: i32) -> ()
        func.func private @addWeightedLine(%in1: memref<1920xui8>, %in2: memref<1920xui8>, %out: memref<1920xui8>, %tileWidth: i32, %alpha: i16, %beta: i16, %gamma: i8) -> ()
        func.func private @thresholdLine(%in: memref<1920xui8>, %out: memref<1920xui8>, %tileWidth: i32, %thresholdValue: i16, %maxValue: i16, %thresholdType: i16) -> ()

        // Declare tile object of the AIE class located at position col 1, row 4
        %tile00 = AIE.tile(0, 0)
        %tile01 = AIE.tile(0, 1)
        %tile02 = AIE.tile(0, 2)
        %tile03 = AIE.tile(0, 3)
        %tile04 = AIE.tile(0, 4)
        %tile05 = AIE.tile(0, 5)

        // Run-time parameter buffers
        %rtp04 = AIE.buffer(%tile04) {sym_name = "rtp04"} : memref<16xi32>
        %rtp05 = AIE.buffer(%tile05) {sym_name = "rtp05"} : memref<16xi32>

        // Declare in and out object FIFOs
        AIE.objectFifo @inOF_L3L1(%tile00, {%tile02}, 2 : i32) : !AIE.objectFifo<memref<7680xui8>>
        
        AIE.objectFifo @outOFL2L3(%tile01, {%tile00}, 2 : i32) : !AIE.objectFifo<memref<1920xui8>>
        AIE.objectFifo @outOFL1L2(%tile05, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<1920xui8>>
        AIE.objectFifo.link [@outOFL1L2] -> [@outOFL2L3] ()

        // Declare task to task object FIFOs

        AIE.objectFifo @OF_2to34(%tile02, {%tile03,%tile04}, [2,2,4])  : !AIE.objectFifo<memref<1920xui8>>
        AIE.objectFifo @OF_3to4(%tile03, {%tile04}, 2 : i32) : !AIE.objectFifo<memref<1920xui8>>
        AIE.objectFifo @OF_4to5(%tile04, {%tile05}, 2 : i32) : !AIE.objectFifo<memref<1920xui8>>
       
        // Define the algorithm for the core of tile(0,2) 
        %core02 = AIE.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %tileHeight = arith.constant 1080 : index
            %tileWidth  = arith.constant 1920 : i32
            %intmax = arith.constant 0xFFFFFFFF : index
            
            scf.for %iter = %c0 to %intmax step %c1 { 
                // Acquire objectFifos and get subviews
                %subviewIn = AIE.objectFifo.acquire @inOF_L3L1(Consume, 1) : !AIE.objectFifoSubview<memref<7680xui8>>
                %elemIn = AIE.objectFifo.subview.access %subviewIn[0] : !AIE.objectFifoSubview<memref<7680xui8>> -> memref<7680xui8>
                %subviewOut = AIE.objectFifo.acquire @OF_2to34(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemOut = AIE.objectFifo.subview.access %subviewOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>

                func.call @rgba2grayLine(%elemIn, %elemOut, %tileWidth) : (memref<7680xui8>, memref<1920xui8>, i32) -> ()

                // Release objectFifos
                AIE.objectFifo.release @inOF_L3L1(Consume, 1)
                AIE.objectFifo.release @OF_2to34(Produce, 1)
            }

            AIE.end
        } { link_with="rgba2gray.cc.o" } // indicate kernel object name used by this core

        // Define the algorithm for the core of tile(0,3) 
        %core03 = AIE.core(%tile03) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %tileHeight       = arith.constant 1080  : index
            %tileWidth        = arith.constant 1920 : i32
            %intmax           = arith.constant 0xFFFFFFFF : index

            scf.for %rep = %c0 to %intmax step %c1 {
                // Acquire objectFifos and get subviews
                %subviewIn = AIE.objectFifo.acquire @OF_2to34(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elem0 = AIE.objectFifo.subview.access %subviewIn[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewOut = AIE.objectFifo.acquire @OF_3to4(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemOut = AIE.objectFifo.subview.access %subviewOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>

                func.call @median1DLine(%elem0, %elemOut, %tileWidth) : (memref<1920xui8>, memref<1920xui8>, i32) -> ()

                // Release objectFifos
                AIE.objectFifo.release @OF_2to34(Consume, 1)
                AIE.objectFifo.release @OF_3to4(Produce, 1)
            
            }
             
            AIE.end
        } { link_with="median.cc.o" } // indicate kernel object name used by this core


        // Define the algorithm for the core of tile(0,4) 
        %core04 = AIE.core(%tile04) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %tileHeight = arith.constant 1080  : index
            %tileWidth  = arith.constant 1920 : i32
            %intmax     = arith.constant 0xFFFFFFFF : index
            
            scf.for %iter = %c0 to %intmax step %c1 { 
                // Acquire objectFifos and get subviews
                %subviewIn0 = AIE.objectFifo.acquire @OF_2to34(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemIn0 = AIE.objectFifo.subview.access %subviewIn0[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewIn1 = AIE.objectFifo.acquire @OF_3to4(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemIn1 = AIE.objectFifo.subview.access %subviewIn1[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewOut = AIE.objectFifo.acquire @OF_4to5(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemOut = AIE.objectFifo.subview.access %subviewOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>

                %a     = memref.load %rtp04[%c0] : memref<16xi32>
                %b     = memref.load %rtp04[%c1] : memref<16xi32>
                %g     = memref.load %rtp04[%c2] : memref<16xi32>
                %alpha = arith.trunci         %a : i32 to i16
                %beta  = arith.trunci         %b : i32 to i16
                %gamma = arith.trunci         %g : i32 to  i8

                func.call @addWeightedLine(%elemIn0, %elemIn1, %elemOut, %tileWidth, %alpha, %beta, %gamma) : (memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i8) -> ()

                // Release objectFifos
                AIE.objectFifo.release @OF_2to34(Consume, 1)
                AIE.objectFifo.release @OF_3to4(Consume, 1)
                AIE.objectFifo.release @OF_4to5(Produce, 1)
            }
            AIE.end
        } { link_with="addWeighted.cc.o" } // indicate kernel object name used by this core

        // Define the algorithm for the core of tile(0,5) 
        %core05 = AIE.core(%tile05) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %tileHeight       = arith.constant 1080  : index
            %tileHeightMinus1 = arith.constant 1079  : index
            %tileWidth        = arith.constant 1920 : i32
            %intmax = arith.constant 0xFFFFFFFF : index

            scf.for %rep = %c0 to %intmax step %c1 {
                // Acquire objectFifos and get subviews
                %subviewIn = AIE.objectFifo.acquire @OF_4to5(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elem0 = AIE.objectFifo.subview.access %subviewIn[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewOut = AIE.objectFifo.acquire @outOFL1L2(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemOut = AIE.objectFifo.subview.access %subviewOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>

                %tv                  = memref.load %rtp05[%c0] : memref<16xi32>
                %tmb                 = memref.load %rtp05[%c1] : memref<16xi32>
                %mv                  = memref.load %rtp05[%c2] : memref<16xi32>
                %thresholdValue      = arith.trunci        %tv : i32 to i16
                %thresholdModeBinary = arith.trunci       %tmb : i32 to i16
                %maxValue            = arith.trunci        %mv : i32 to i16

                //func.call @dilate1DLine(%elem0, %elemOut, %tileWidth) : (memref<1920xui8>, memref<1920xui8>, i32) -> ()
                func.call @thresholdLine(%elem0, %elemOut, %tileWidth, %thresholdValue, %maxValue, %thresholdModeBinary) : (memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i16) -> ()

                // Release objectFifos
                AIE.objectFifo.release @OF_4to5(Consume, 1)
                AIE.objectFifo.release @outOFL1L2(Produce, 1)
            }

            AIE.end
        } { link_with="threshold.cc.o" } // indicate kernel object name used by this core

        func.func @sequence(%in : memref<1080x1920xi32>, %out : memref<1080x480xi32>) {
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i32
            %tileHeight     = arith.constant    1080 : i32
            %tileWidthRGBA  = arith.constant    1920 : i32  // in 32b words so tileWidth (since there are 4 channels in rgba)
            %tileWidthGray  = arith.constant     480 : i32  // in 32b words so tileWidth/4 (since there is only 1 channels in gray)
            %totalLenRGBA   = arith.constant 2073600 : i32
            %totalLenGray   = arith.constant  518400 : i32

                                                                                   //                              noise out         denoise out
            AIEX.ipu.rtp_write(0, 4, 0,      16384) { buffer_sym_name = "rtp04" }  // alpha                  16384 = 1 << 14                   0                  
            AIEX.ipu.rtp_write(0, 4, 1, 0xFFFFC000) { buffer_sym_name = "rtp04" }  // beta                 -16384 = -1 << 14     16384 = 1 << 14 
            AIEX.ipu.rtp_write(0, 4, 2,          0) { buffer_sym_name = "rtp04" }  // gamma                                0                   0              
            AIEX.ipu.rtp_write(0, 5, 0,         10) { buffer_sym_name = "rtp05" }  // thresholdValue                      10                   0
            AIEX.ipu.rtp_write(0, 5, 1,          0) { buffer_sym_name = "rtp05" }  // thresholdModeBinary         binary = 0          toZero = 3
            AIEX.ipu.rtp_write(0, 5, 2,        255) { buffer_sym_name = "rtp05" }  // maxValue                           255                 255

            //dma_memcpy_nd ([offset in 32b words][length in 32b words][stride in 32b words])
            AIEX.ipu.dma_memcpy_nd (%c0, %c0, %in [%c0, %c0, %c0, %c0][%c1, %c1, %c1, %totalLenRGBA][%c0, %c0, %c0]) { metadata = @inOF_L3L1,  id = 1 : i32 } : (i32, i32, memref<1080x1920xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
            AIEX.ipu.dma_memcpy_nd (%c0, %c0, %out[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %totalLenGray][%c0, %c0, %c0]) { metadata = @outOFL2L3, id = 0 : i32 } : (i32, i32, memref<1080x480xi32>,  [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
            AIEX.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
            return
        }
    }
}
