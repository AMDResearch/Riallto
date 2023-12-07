// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// Declare this MLIR module. A block encapsulates all 
// AIE tiles, buffers, and communication in an AI Engine design
module @colorDetect_aie2 {

 	AIE.device(ipu) {
        // declare kernel external kernel function 
        func.func private @rgba2hueLine(%in: memref<5120xui8>, %out: memref<1280xui8>, %tileWidth: i32) -> ()
        func.func private @thresholdLine(%in: memref<1280xui8>, %out: memref<1280xui8>, %tileWidth: i32, %thresholdValue: i16, %maxValue: i16, %thresholdType: i16) -> ()
        func.func private @bitwiseORLine(%in1: memref<1280xui8>, %in2: memref<1280xui8>, %out: memref<1280xui8>, %lineWidth: i32) -> ()
        func.func private @gray2rgbaLine(%in: memref<1280xui8>, %out: memref<5120xui8>, %tileWidth: i32) -> ()
        func.func private @bitwiseANDLine(%in1: memref<5120xui8>, %in2: memref<5120xui8>, %out: memref<5120xui8>, %lineWidth: i32) -> ()

        // Declare tile object of the AIE class located at position col 1, row 4
        %tile00 = AIE.tile(0, 0)
        %tile01 = AIE.tile(0, 1)
        %tile02 = AIE.tile(0, 2)
        %tile03 = AIE.tile(0, 3)
        %tile04 = AIE.tile(0, 4)
        %tile05 = AIE.tile(0, 5)

        // Run-time parameters
    	%rtp0 = AIE.buffer(%tile02) {sym_name = "rtp0"} : memref<16xi32>
    	%rtp1 = AIE.buffer(%tile03) {sym_name = "rtp1"} : memref<16xi32>
    	%rtp2 = AIE.buffer(%tile04) {sym_name = "rtp2"} : memref<16xi32>
    	%rtp3 = AIE.buffer(%tile05) {sym_name = "rtp3"} : memref<16xi32>

        // Declare in and out object FIFOs
        AIE.objectFifo @inOF_L3L1(%tile00, {%tile02, %tile05}, [2,2,6]) : !AIE.objectFifo<memref<5120xui8>>
        
        AIE.objectFifo @outOFL2L3(%tile01, {%tile00}, 2 : i32) : !AIE.objectFifo<memref<5120xui8>>
        AIE.objectFifo @outOFL1L2(%tile05, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<5120xui8>>
        AIE.objectFifo.link [@outOFL1L2] -> [@outOFL2L3] ()

        // Declare task to task object FIFOs
        AIE.objectFifo @OF_2to34(%tile02, {%tile03, %tile04}, 2 : i32) : !AIE.objectFifo<memref<1280xui8>>

        AIE.objectFifo @OF_3to3(%tile03, {%tile03}, 1 : i32) : !AIE.objectFifo<memref<1280xui8>>
        AIE.objectFifo @OF_3to5(%tile03, {%tile05}, 2 : i32) : !AIE.objectFifo<memref<1280xui8>>

        AIE.objectFifo @OF_4to4(%tile04, {%tile04}, 1 : i32) : !AIE.objectFifo<memref<1280xui8>>
        AIE.objectFifo @OF_4to5(%tile04, {%tile05}, 2 : i32) : !AIE.objectFifo<memref<1280xui8>>

        AIE.objectFifo @OF_5to5(%tile05, {%tile05}, 1 : i32) : !AIE.objectFifo<memref<1280xui8>>
        AIE.objectFifo @OF_5to5_2(%tile05, {%tile05}, 1 : i32) : !AIE.objectFifo<memref<5120xui8>>
       
        // Define the algorithm for the core of tile(0,2) 
        %core02 = AIE.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %tileHeight = arith.constant 720  : index
            %tileWidth  = arith.constant 1280 : i32
            %intmax = arith.constant 0xFFFFFFFF : index
            
            scf.for %iter = %c0 to %intmax step %c1 { 
                // Acquire objectFifos and get subviews
                %subviewIn = AIE.objectFifo.acquire @inOF_L3L1(Consume, 1) : !AIE.objectFifoSubview<memref<5120xui8>>
                %elemIn = AIE.objectFifo.subview.access %subviewIn[0] : !AIE.objectFifoSubview<memref<5120xui8>> -> memref<5120xui8>
                %subviewOut = AIE.objectFifo.acquire @OF_2to34(Produce, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemOut = AIE.objectFifo.subview.access %subviewOut[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>

                func.call @rgba2hueLine(%elemIn, %elemOut, %tileWidth) : (memref<5120xui8>, memref<1280xui8>, i32) -> ()

                // Release objectFifos
                AIE.objectFifo.release @inOF_L3L1(Consume, 1)
                AIE.objectFifo.release @OF_2to34(Produce, 1)
            }
            AIE.end
        } { link_with="rgba2hue.cc.o" } // indicate kernel object name used by this core

        // Define the algorithm for the core of tile(0,4) 
        %core03 = AIE.core(%tile03) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c3 = arith.constant 3 : index
            %tileHeight = arith.constant 720  : index
            %tileWidth  = arith.constant 1280 : i32
            %thresholdModeToZero = arith.constant 3 : i16
            %thresholdModeBinaryInv =  arith.constant 1 : i16
            %maxValue        = arith.constant 255 : i16
            %intmax = arith.constant 0xFFFFFFFF : index
            
            scf.for %iter = %c0 to %intmax step %c1 { 
                // Acquire objectFifos and get subviews
                %subviewIn = AIE.objectFifo.acquire @OF_2to34(Consume, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemIn = AIE.objectFifo.subview.access %subviewIn[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>
                %subviewOut = AIE.objectFifo.acquire @OF_3to3(Produce, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemOut = AIE.objectFifo.subview.access %subviewOut[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>

	    	%thresh_u = memref.load %rtp1[%c0] : memref<16xi32>
	    	%thresholdValue1u = arith.trunci %thresh_u : i32 to i16 
	    	%thresh_l = memref.load %rtp1[%c1] : memref<16xi32>
	    	%thresholdValue1l = arith.trunci %thresh_l : i32 to i16 

                func.call @thresholdLine(%elemIn, %elemOut, %tileWidth, %thresholdValue1u, %maxValue, %thresholdModeToZero) : (memref<1280xui8>, memref<1280xui8>, i32, i16, i16, i16) -> ()

                // Release objectFifos
                AIE.objectFifo.release @OF_2to34(Consume, 1)
                AIE.objectFifo.release @OF_3to3(Produce, 1)
            
                // Acquire objectFifos and get subviews
                %subviewIn1 = AIE.objectFifo.acquire @OF_3to3(Consume, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemIn1 = AIE.objectFifo.subview.access %subviewIn1[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>
                %subviewOut1 = AIE.objectFifo.acquire @OF_3to5(Produce, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemOut1 = AIE.objectFifo.subview.access %subviewOut1[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>

                func.call @thresholdLine(%elemIn1, %elemOut1, %tileWidth, %thresholdValue1l, %maxValue, %thresholdModeBinaryInv) : (memref<1280xui8>, memref<1280xui8>, i32, i16, i16, i16) -> ()

                // Release objectFifos
                AIE.objectFifo.release @OF_3to3(Consume, 1)
                AIE.objectFifo.release @OF_3to5(Produce, 1)
            }

            AIE.end
        } { link_with="threshold.cc.o" } // indicate kernel object name used by this core

        // Define the algorithm for the core of tile(0,4) 
        %core04 = AIE.core(%tile04) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %tileHeight = arith.constant 720  : index
            %tileWidth  = arith.constant 1280 : i32
            %thresholdModeToZero = arith.constant 3 : i16
            %thresholdModeBinaryInv =  arith.constant 1 : i16
            %maxValue        = arith.constant 255 : i16
            %intmax = arith.constant 0xFFFFFFFF : index
            
            scf.for %iter = %c0 to %intmax step %c1 { 
                // Acquire objectFifos and get subviews
                %subviewIn = AIE.objectFifo.acquire @OF_2to34(Consume, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemIn = AIE.objectFifo.subview.access %subviewIn[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>
                %subviewOut = AIE.objectFifo.acquire @OF_4to4(Produce, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemOut = AIE.objectFifo.subview.access %subviewOut[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>

	    	%thresh_u = memref.load %rtp2[%c0] : memref<16xi32>
	    	%thresholdValue2u = arith.trunci %thresh_u : i32 to i16 
	    	%thresh_l = memref.load %rtp2[%c1] : memref<16xi32>
	    	%thresholdValue2l = arith.trunci %thresh_l : i32 to i16 


                func.call @thresholdLine(%elemIn, %elemOut, %tileWidth, %thresholdValue2u, %maxValue, %thresholdModeToZero) : (memref<1280xui8>, memref<1280xui8>, i32, i16, i16, i16) -> ()

                // Release objectFifos
                AIE.objectFifo.release @OF_2to34(Consume, 1)
                AIE.objectFifo.release @OF_4to4(Produce, 1)
             
                // Acquire objectFifos and get subviews
                %subviewIn1 = AIE.objectFifo.acquire @OF_4to4(Consume, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemIn1 = AIE.objectFifo.subview.access %subviewIn1[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>
                %subviewOut1 = AIE.objectFifo.acquire @OF_4to5(Produce, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemOut1 = AIE.objectFifo.subview.access %subviewOut1[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>

                func.call @thresholdLine(%elemIn1, %elemOut1, %tileWidth, %thresholdValue2l, %maxValue, %thresholdModeBinaryInv) : (memref<1280xui8>, memref<1280xui8>, i32, i16, i16, i16) -> ()

                // Release objectFifos
                AIE.objectFifo.release @OF_4to4(Consume, 1)
                AIE.objectFifo.release @OF_4to5(Produce, 1)
            }

            AIE.end
        } { link_with="threshold.cc.o" } // indicate kernel object name used by this core    

        // Define the algorithm for the core of tile(0,5) 
        %core05 = AIE.core(%tile05) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %tileHeight = arith.constant 720  : index
            %tileWidth  = arith.constant 1280 : i32
            %tileWidthRGBA = arith.constant 5120 : i32
            %intmax = arith.constant 0xFFFFFFFF : index
            
            scf.for %iter = %c0 to %intmax step %c1 { 
                // Acquire objectFifos and get subviews
                %subviewIn0 = AIE.objectFifo.acquire @OF_3to5(Consume, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemIn0 = AIE.objectFifo.subview.access %subviewIn0[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>
                %subviewIn1 = AIE.objectFifo.acquire @OF_4to5(Consume, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemIn1 = AIE.objectFifo.subview.access %subviewIn1[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>
                %subviewOut0 = AIE.objectFifo.acquire @OF_5to5(Produce, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemOut0 = AIE.objectFifo.subview.access %subviewOut0[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>

                func.call @bitwiseORLine(%elemIn0, %elemIn1, %elemOut0, %tileWidth) :  (memref<1280xui8>, memref<1280xui8>, memref<1280xui8>, i32) -> ()

                 // Release objectFifos
                AIE.objectFifo.release @OF_3to5(Consume, 1)
                AIE.objectFifo.release @OF_4to5(Consume, 1)
                AIE.objectFifo.release @OF_5to5(Produce, 1)
                
                // 2 kernel
                // Acquire objectFifos and get subviews
                %subviewIn2 = AIE.objectFifo.acquire @OF_5to5(Consume, 1) : !AIE.objectFifoSubview<memref<1280xui8>>
                %elemIn2 = AIE.objectFifo.subview.access %subviewIn2[0] : !AIE.objectFifoSubview<memref<1280xui8>> -> memref<1280xui8>
                %subviewOut1 = AIE.objectFifo.acquire @OF_5to5_2(Produce, 1) : !AIE.objectFifoSubview<memref<5120xui8>>
                %elemOut1 = AIE.objectFifo.subview.access %subviewOut1[0] : !AIE.objectFifoSubview<memref<5120xui8>> -> memref<5120xui8>

                func.call @gray2rgbaLine(%elemIn2, %elemOut1, %tileWidth) : (memref<1280xui8>, memref<5120xui8>, i32) -> ()

                // Release objectFifos
                AIE.objectFifo.release @OF_5to5(Consume, 1)
                AIE.objectFifo.release @OF_5to5_2(Produce, 1)

                // 3 kernel
                // Acquire objectFifos and get subviews
                %subviewIn3 = AIE.objectFifo.acquire @OF_5to5_2(Consume, 1) : !AIE.objectFifoSubview<memref<5120xui8>>
                %elemIn3 = AIE.objectFifo.subview.access %subviewIn3[0] : !AIE.objectFifoSubview<memref<5120xui8>> -> memref<5120xui8>
                %subviewIn4 = AIE.objectFifo.acquire @inOF_L3L1(Consume, 1) : !AIE.objectFifoSubview<memref<5120xui8>>
                %elemIn4 = AIE.objectFifo.subview.access %subviewIn4[0] : !AIE.objectFifoSubview<memref<5120xui8>> -> memref<5120xui8>
                %subviewOut2 = AIE.objectFifo.acquire @outOFL1L2(Produce, 1) : !AIE.objectFifoSubview<memref<5120xui8>>
                %elemOut2 = AIE.objectFifo.subview.access %subviewOut2[0] : !AIE.objectFifoSubview<memref<5120xui8>> -> memref<5120xui8>

                func.call @bitwiseANDLine(%elemIn3, %elemIn4, %elemOut2, %tileWidthRGBA) : (memref<5120xui8>, memref<5120xui8>, memref<5120xui8>, i32) -> ()

                // Release objectFifos
                AIE.objectFifo.release @OF_5to5_2(Consume, 1)
                AIE.objectFifo.release @inOF_L3L1(Consume, 1)
                AIE.objectFifo.release @outOFL1L2(Produce, 1)
            }
            AIE.end
        } { link_with="combined_bitwiseOR_gray2rgba_bitwiseAND.a" } // indicate kernel object name used by this core

        func.func @sequence(%in : memref<720x1280xi32>, %out : memref<720x1280xi32>) {
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i32
            %tileHeight = arith.constant 720  : i32
            %tileWidth  = arith.constant 1280 : i32  // in 32b words so tileWidth
            %totalLenRGBA = arith.constant 921600 : i32

            AIEX.ipu.rtp_write(0, 3, 0, 0) { buffer_sym_name = "rtp1" } // thresholdValue1u
            AIEX.ipu.rtp_write(0, 3, 1, 1) { buffer_sym_name = "rtp1" } // thresholdValue1l

            AIEX.ipu.rtp_write(0, 4, 0, 0) { buffer_sym_name = "rtp2" } // thresholdValue2u
            AIEX.ipu.rtp_write(0, 4, 1, 1) { buffer_sym_name = "rtp2" } // thresholdValue2l

            //dma_memcpy_nd ([offset in 32b words][length in 32b words][stride in 32b words])
            AIEX.ipu.dma_memcpy_nd (%c0, %c0, %in [%c0, %c0, %c0, %c0][%c1, %c1, %c1, %totalLenRGBA][%c0, %c0, %c0]) { metadata = @inOF_L3L1, id = 1 : i32 }  : (i32, i32, memref<720x1280xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
            AIEX.ipu.dma_memcpy_nd (%c0, %c0, %out[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %totalLenRGBA][%c0, %c0, %c0]) { metadata = @outOFL2L3, id = 0 : i32 } : (i32, i32, memref<720x1280xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
            AIEX.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
            return
        }
    }
}
