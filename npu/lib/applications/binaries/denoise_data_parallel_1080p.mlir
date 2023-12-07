// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// Declare this MLIR module. A block encapsulates all 
// AIE tiles, buffers, and communication in an AI Engine design
module @denoise_aie2 {

 	AIE.device(ipu) {
        // declare kernel external kernel function 
        func.func private @rgba2grayLine(%in: memref<7680xui8>, %out: memref<1920xui8>, %tileWidth: i32) -> ()
        func.func private @median1DLine(%in: memref<1920xui8>, %out: memref<1920xui8>, %lineWidth: i32) -> ()
        func.func private @addWeightedLine(%in1: memref<1920xui8>, %in1: memref<1920xui8>, %out: memref<1920xui8>, %tileWidth: i32, %alpha: i16, %beta: i16, %gamma: i8) -> ()
        func.func private @thresholdLine(%in: memref<1920xui8>, %out: memref<1920xui8>, %tileWidth: i32, %thresholdValue: i16, %maxValue: i16, %thresholdType: i16) -> ()

        // Declare tile object of the AIE class located at position col 1, row 4
        %tile00 = AIE.tile(0, 0)
        %tile01 = AIE.tile(0, 1)
        %tile02 = AIE.tile(0, 2)
        %tile03 = AIE.tile(0, 3)
        %tile04 = AIE.tile(0, 4)
        %tile05 = AIE.tile(0, 5)

        // Run-time parameter buffers
        %rtp02 = AIE.buffer(%tile02) {sym_name = "rtp02"} : memref<16xi32>
        %rtp03 = AIE.buffer(%tile03) {sym_name = "rtp03"} : memref<16xi32>
        %rtp04 = AIE.buffer(%tile04) {sym_name = "rtp04"} : memref<16xi32>
        %rtp05 = AIE.buffer(%tile05) {sym_name = "rtp05"} : memref<16xi32>

        // Declare in and out object FIFOs
        AIE.objectFifo @inOF_L3L2(%tile00, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<30720xui8>>
        AIE.objectFifo @inOF_L2L1_2(%tile01, {%tile02}, 2 : i32) : !AIE.objectFifo<memref<7680xui8>>
        AIE.objectFifo @inOF_L2L1_3(%tile01, {%tile03}, 2 : i32) : !AIE.objectFifo<memref<7680xui8>>
        AIE.objectFifo @inOF_L2L1_4(%tile01, {%tile04}, 2 : i32) : !AIE.objectFifo<memref<7680xui8>>
        AIE.objectFifo @inOF_L2L1_5(%tile01, {%tile05}, 2 : i32) : !AIE.objectFifo<memref<7680xui8>>
        AIE.objectFifo.link [@inOF_L3L2] -> [@inOF_L2L1_2, @inOF_L2L1_3, @inOF_L2L1_4, @inOF_L2L1_5] ()
        
        AIE.objectFifo @outOFL1L2_2(%tile02, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<1920xui8>>
        AIE.objectFifo @outOFL1L2_3(%tile03, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<1920xui8>>
        AIE.objectFifo @outOFL1L2_4(%tile04, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<1920xui8>>
        AIE.objectFifo @outOFL1L2_5(%tile05, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<1920xui8>>
        AIE.objectFifo @outOFL2L3(%tile01, {%tile00}, 2 : i32) : !AIE.objectFifo<memref<7680xui8>>
        AIE.objectFifo.link [@outOFL1L2_2, @outOFL1L2_3, @outOFL1L2_4, @outOFL1L2_5] -> [@outOFL2L3] ()

        // Declare task to task object FIFOs
        AIE.objectFifo @OF_rgba_out_2(%tile02, {%tile02}, 1 : i32) : !AIE.objectFifo<memref<1920xui8>>
        AIE.objectFifo @OF_median_out_2(%tile02, {%tile02}, 1 : i32) : !AIE.objectFifo<memref<1920xui8>>
        AIE.objectFifo @OF_add_out_2(%tile02, {%tile02}, 1 : i32) : !AIE.objectFifo<memref<1920xui8>>

        AIE.objectFifo @OF_rgba_out_3(%tile03, {%tile03}, 1 : i32) : !AIE.objectFifo<memref<1920xui8>>
        AIE.objectFifo @OF_median_out_3(%tile03, {%tile03}, 1 : i32) : !AIE.objectFifo<memref<1920xui8>>
        AIE.objectFifo @OF_add_out_3(%tile03, {%tile03}, 1 : i32) : !AIE.objectFifo<memref<1920xui8>>

        AIE.objectFifo @OF_rgba_out_4(%tile04, {%tile04}, 1 : i32) : !AIE.objectFifo<memref<1920xui8>>
        AIE.objectFifo @OF_median_out_4(%tile04, {%tile04}, 1 : i32) : !AIE.objectFifo<memref<1920xui8>>
        AIE.objectFifo @OF_add_out_4(%tile04, {%tile04}, 1 : i32) : !AIE.objectFifo<memref<1920xui8>>

        AIE.objectFifo @OF_rgba_out_5(%tile05, {%tile05}, 1 : i32) : !AIE.objectFifo<memref<1920xui8>>
        AIE.objectFifo @OF_median_out_5(%tile05, {%tile05}, 1 : i32) : !AIE.objectFifo<memref<1920xui8>>
        AIE.objectFifo @OF_add_out_5(%tile05, {%tile05}, 1 : i32) : !AIE.objectFifo<memref<1920xui8>>
       
        // Define the algorithm for the core of tile(0,2) 
        %core02 = AIE.core(%tile02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c3 = arith.constant 3 : index
            %c4 = arith.constant 4 : index
            %c5 = arith.constant 5 : index
            %tileHeight = arith.constant  270  : index
            %tileWidth  = arith.constant 1920 : i32
            %intmax     = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %intmax step %c1 { 
                // Acquire objectFifos and get subviews
                %subviewRGBAIn = AIE.objectFifo.acquire @inOF_L2L1_2(Consume, 1) : !AIE.objectFifoSubview<memref<7680xui8>>
                %elemRGBAIn = AIE.objectFifo.subview.access %subviewRGBAIn[0] : !AIE.objectFifoSubview<memref<7680xui8>> -> memref<7680xui8>
                %subviewRGBAOut = AIE.objectFifo.acquire @OF_rgba_out_2(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemRGBAOut = AIE.objectFifo.subview.access %subviewRGBAOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                func.call @rgba2grayLine(%elemRGBAIn, %elemRGBAOut, %tileWidth) : (memref<7680xui8>, memref<1920xui8>, i32) -> ()
                AIE.objectFifo.release @inOF_L2L1_2(Consume, 1)
                AIE.objectFifo.release @OF_rgba_out_2(Produce, 1)

                %subviewMedianIn = AIE.objectFifo.acquire @OF_rgba_out_2(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemMedian0 = AIE.objectFifo.subview.access %subviewMedianIn[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewMedianOut = AIE.objectFifo.acquire @OF_median_out_2(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemMedianOut = AIE.objectFifo.subview.access %subviewMedianOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                func.call @median1DLine(%elemMedian0, %elemMedianOut, %tileWidth) : (memref<1920xui8>, memref<1920xui8>, i32) -> ()
                AIE.objectFifo.release @OF_median_out_2(Produce, 1)

                %subviewAddWeightedIn0 = AIE.objectFifo.acquire @OF_rgba_out_2(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemAddWeightedIn0 = AIE.objectFifo.subview.access %subviewAddWeightedIn0[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewAddWeightedIn1 = AIE.objectFifo.acquire @OF_median_out_2(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemAddWeightedIn1 = AIE.objectFifo.subview.access %subviewAddWeightedIn1[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewAddWeightedOut = AIE.objectFifo.acquire @OF_add_out_2(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemAddWeightedOut = AIE.objectFifo.subview.access %subviewAddWeightedOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %a     = memref.load %rtp02[%c0] : memref<16xi32>
                %b     = memref.load %rtp02[%c1] : memref<16xi32>
                %g     = memref.load %rtp02[%c2] : memref<16xi32>
                %alpha = arith.trunci         %a : i32 to i16
                %beta  = arith.trunci         %b : i32 to i16
                %gamma = arith.trunci         %g : i32 to  i8
                func.call @addWeightedLine(%elemAddWeightedIn0, %elemAddWeightedIn1, %elemAddWeightedOut, %tileWidth, %alpha, %beta, %gamma) : (memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i8) -> ()
                AIE.objectFifo.release @OF_rgba_out_2(Consume, 1)
                AIE.objectFifo.release @OF_median_out_2(Consume, 1)
                AIE.objectFifo.release @OF_add_out_2(Produce, 1)               

                %subviewThresholdIn = AIE.objectFifo.acquire @OF_add_out_2(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemThreshold0 = AIE.objectFifo.subview.access %subviewThresholdIn[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewThresholdOut = AIE.objectFifo.acquire @outOFL1L2_2(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemThresholdOut = AIE.objectFifo.subview.access %subviewThresholdOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %tv                  = memref.load %rtp02[%c3] : memref<16xi32>
                %tmb                 = memref.load %rtp02[%c4] : memref<16xi32>
                %mv                  = memref.load %rtp02[%c5] : memref<16xi32>
                %thresholdValue      = arith.trunci        %tv : i32 to i16
                %thresholdModeBinary = arith.trunci       %tmb : i32 to i16
                %maxValue            = arith.trunci        %mv : i32 to i16
                func.call @thresholdLine(%elemThreshold0, %elemThresholdOut, %tileWidth, %thresholdValue, %maxValue, %thresholdModeBinary) : (memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i16) -> ()
                AIE.objectFifo.release @OF_add_out_2(Consume, 1)
                AIE.objectFifo.release @outOFL1L2_2(Produce, 1)
            }

            AIE.end
        } { link_with="combined_rgba2gray_median_addWeighted_threshold.a" } // indicate kernel object name used by this core

        // Define the algorithm for the core of tile(0,3) 
        %core03 = AIE.core(%tile03) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c3 = arith.constant 3 : index
            %c4 = arith.constant 4 : index
            %c5 = arith.constant 5 : index
            %tileHeight = arith.constant  270 : index
            %tileWidth  = arith.constant 1920 : i32
            %intmax     = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %intmax step %c1 { 
                // Acquire objectFifos and get subviews
                %subviewRGBAIn = AIE.objectFifo.acquire @inOF_L2L1_3(Consume, 1) : !AIE.objectFifoSubview<memref<7680xui8>>
                %elemRGBAIn = AIE.objectFifo.subview.access %subviewRGBAIn[0] : !AIE.objectFifoSubview<memref<7680xui8>> -> memref<7680xui8>
                %subviewRGBAOut = AIE.objectFifo.acquire @OF_rgba_out_3(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemRGBAOut = AIE.objectFifo.subview.access %subviewRGBAOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                func.call @rgba2grayLine(%elemRGBAIn, %elemRGBAOut, %tileWidth) : (memref<7680xui8>, memref<1920xui8>, i32) -> ()
                AIE.objectFifo.release @inOF_L2L1_3(Consume, 1)
                AIE.objectFifo.release @OF_rgba_out_3(Produce, 1)

                %subviewMedianIn = AIE.objectFifo.acquire @OF_rgba_out_3(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemMedian0 = AIE.objectFifo.subview.access %subviewMedianIn[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewMedianOut = AIE.objectFifo.acquire @OF_median_out_3(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemMedianOut = AIE.objectFifo.subview.access %subviewMedianOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                func.call @median1DLine(%elemMedian0, %elemMedianOut, %tileWidth) : (memref<1920xui8>, memref<1920xui8>, i32) -> ()
                AIE.objectFifo.release @OF_median_out_3(Produce, 1)

                %subviewAddWeightedIn0 = AIE.objectFifo.acquire @OF_rgba_out_3(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemAddWeightedIn0 = AIE.objectFifo.subview.access %subviewAddWeightedIn0[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewAddWeightedIn1 = AIE.objectFifo.acquire @OF_median_out_3(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemAddWeightedIn1 = AIE.objectFifo.subview.access %subviewAddWeightedIn1[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewAddWeightedOut = AIE.objectFifo.acquire @OF_add_out_3(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemAddWeightedOut = AIE.objectFifo.subview.access %subviewAddWeightedOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %a     = memref.load %rtp03[%c0] : memref<16xi32>
                %b     = memref.load %rtp03[%c1] : memref<16xi32>
                %g     = memref.load %rtp03[%c2] : memref<16xi32>
                %alpha = arith.trunci         %a : i32 to i16
                %beta  = arith.trunci         %b : i32 to i16
                %gamma = arith.trunci         %g : i32 to  i8
                func.call @addWeightedLine(%elemAddWeightedIn0, %elemAddWeightedIn1, %elemAddWeightedOut, %tileWidth, %alpha, %beta, %gamma) : (memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i8) -> ()
                AIE.objectFifo.release @OF_rgba_out_3(Consume, 1)
                AIE.objectFifo.release @OF_median_out_3(Consume, 1)
                AIE.objectFifo.release @OF_add_out_3(Produce, 1)               

                %subviewThresholdIn = AIE.objectFifo.acquire @OF_add_out_3(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemThreshold0 = AIE.objectFifo.subview.access %subviewThresholdIn[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewThresholdOut = AIE.objectFifo.acquire @outOFL1L2_3(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemThresholdOut = AIE.objectFifo.subview.access %subviewThresholdOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %tv                  = memref.load %rtp03[%c3] : memref<16xi32>
                %tmb                 = memref.load %rtp03[%c4] : memref<16xi32>
                %mv                  = memref.load %rtp03[%c5] : memref<16xi32>
                %thresholdValue      = arith.trunci        %tv : i32 to i16
                %thresholdModeBinary = arith.trunci       %tmb : i32 to i16
                %maxValue            = arith.trunci        %mv : i32 to i16
                func.call @thresholdLine(%elemThreshold0, %elemThresholdOut, %tileWidth, %thresholdValue, %maxValue, %thresholdModeBinary) : (memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i16) -> ()
                AIE.objectFifo.release @OF_add_out_3(Consume, 1)
                AIE.objectFifo.release @outOFL1L2_3(Produce, 1)
            }

            AIE.end
        } { link_with="combined_rgba2gray_median_addWeighted_threshold.a" } // indicate kernel object name used by this core


        // Define the algorithm for the core of tile(0,4) 
        %core04 = AIE.core(%tile04) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c3 = arith.constant 3 : index
            %c4 = arith.constant 4 : index
            %c5 = arith.constant 5 : index
            %tileHeight = arith.constant  270 : index
            %tileWidth  = arith.constant 1920 : i32
            %intmax    = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %intmax step %c1 { 
                // Acquire objectFifos and get subviews
                %subviewRGBAIn = AIE.objectFifo.acquire @inOF_L2L1_4(Consume, 1) : !AIE.objectFifoSubview<memref<7680xui8>>
                %elemRGBAIn = AIE.objectFifo.subview.access %subviewRGBAIn[0] : !AIE.objectFifoSubview<memref<7680xui8>> -> memref<7680xui8>
                %subviewRGBAOut = AIE.objectFifo.acquire @OF_rgba_out_4(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemRGBAOut = AIE.objectFifo.subview.access %subviewRGBAOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                func.call @rgba2grayLine(%elemRGBAIn, %elemRGBAOut, %tileWidth) : (memref<7680xui8>, memref<1920xui8>, i32) -> ()
                AIE.objectFifo.release @inOF_L2L1_4(Consume, 1)
                AIE.objectFifo.release @OF_rgba_out_4(Produce, 1)

                %subviewMedianIn = AIE.objectFifo.acquire @OF_rgba_out_4(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemMedian0 = AIE.objectFifo.subview.access %subviewMedianIn[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewMedianOut = AIE.objectFifo.acquire @OF_median_out_4(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemMedianOut = AIE.objectFifo.subview.access %subviewMedianOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                func.call @median1DLine(%elemMedian0, %elemMedianOut, %tileWidth) : (memref<1920xui8>, memref<1920xui8>, i32) -> ()
                AIE.objectFifo.release @OF_median_out_4(Produce, 1)

                %subviewAddWeightedIn0 = AIE.objectFifo.acquire @OF_rgba_out_4(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemAddWeightedIn0 = AIE.objectFifo.subview.access %subviewAddWeightedIn0[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewAddWeightedIn1 = AIE.objectFifo.acquire @OF_median_out_4(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemAddWeightedIn1 = AIE.objectFifo.subview.access %subviewAddWeightedIn1[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewAddWeightedOut = AIE.objectFifo.acquire @OF_add_out_4(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemAddWeightedOut = AIE.objectFifo.subview.access %subviewAddWeightedOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %a     = memref.load %rtp04[%c0] : memref<16xi32>
                %b     = memref.load %rtp04[%c1] : memref<16xi32>
                %g     = memref.load %rtp04[%c2] : memref<16xi32>
                %alpha = arith.trunci         %a : i32 to i16
                %beta  = arith.trunci         %b : i32 to i16
                %gamma = arith.trunci         %g : i32 to  i8
                func.call @addWeightedLine(%elemAddWeightedIn0, %elemAddWeightedIn1, %elemAddWeightedOut, %tileWidth, %alpha, %beta, %gamma) : (memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i8) -> ()
                AIE.objectFifo.release @OF_rgba_out_4(Consume, 1)
                AIE.objectFifo.release @OF_median_out_4(Consume, 1)
                AIE.objectFifo.release @OF_add_out_4(Produce, 1)               

                %subviewThresholdIn = AIE.objectFifo.acquire @OF_add_out_4(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemThreshold0 = AIE.objectFifo.subview.access %subviewThresholdIn[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewThresholdOut = AIE.objectFifo.acquire @outOFL1L2_4(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemThresholdOut = AIE.objectFifo.subview.access %subviewThresholdOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %tv                  = memref.load %rtp04[%c3] : memref<16xi32>
                %tmb                 = memref.load %rtp04[%c4] : memref<16xi32>
                %mv                  = memref.load %rtp04[%c5] : memref<16xi32>
                %thresholdValue      = arith.trunci        %tv : i32 to i16
                %thresholdModeBinary = arith.trunci       %tmb : i32 to i16
                %maxValue            = arith.trunci        %mv : i32 to i16
                func.call @thresholdLine(%elemThreshold0, %elemThresholdOut, %tileWidth, %thresholdValue, %maxValue, %thresholdModeBinary) : (memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i16) -> ()
                AIE.objectFifo.release @OF_add_out_4(Consume, 1)
                AIE.objectFifo.release @outOFL1L2_4(Produce, 1)
            }

            AIE.end
        } { link_with="combined_rgba2gray_median_addWeighted_threshold.a" } // indicate kernel object name used by this core

        // Define the algorithm for the core of tile(0,5) 
        %core05 = AIE.core(%tile05) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c3 = arith.constant 3 : index
            %c4 = arith.constant 4 : index
            %c5 = arith.constant 5 : index
            %tileHeight = arith.constant  270 : index
            %tileWidth  = arith.constant 1920 : i32
            %intmax    = arith.constant 0xFFFFFFFF : index

            scf.for %iter = %c0 to %intmax step %c1 { 
                // Acquire objectFifos and get subviews
                %subviewRGBAIn = AIE.objectFifo.acquire @inOF_L2L1_5(Consume, 1) : !AIE.objectFifoSubview<memref<7680xui8>>
                %elemRGBAIn = AIE.objectFifo.subview.access %subviewRGBAIn[0] : !AIE.objectFifoSubview<memref<7680xui8>> -> memref<7680xui8>
                %subviewRGBAOut = AIE.objectFifo.acquire @OF_rgba_out_5(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemRGBAOut = AIE.objectFifo.subview.access %subviewRGBAOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                func.call @rgba2grayLine(%elemRGBAIn, %elemRGBAOut, %tileWidth) : (memref<7680xui8>, memref<1920xui8>, i32) -> ()
                AIE.objectFifo.release @inOF_L2L1_5(Consume, 1)
                AIE.objectFifo.release @OF_rgba_out_5(Produce, 1)

                %subviewMedianIn = AIE.objectFifo.acquire @OF_rgba_out_5(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemMedian0 = AIE.objectFifo.subview.access %subviewMedianIn[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewMedianOut = AIE.objectFifo.acquire @OF_median_out_5(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemMedianOut = AIE.objectFifo.subview.access %subviewMedianOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                func.call @median1DLine(%elemMedian0, %elemMedianOut, %tileWidth) : (memref<1920xui8>, memref<1920xui8>, i32) -> ()
                AIE.objectFifo.release @OF_median_out_5(Produce, 1)

                %subviewAddWeightedIn0 = AIE.objectFifo.acquire @OF_rgba_out_5(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemAddWeightedIn0 = AIE.objectFifo.subview.access %subviewAddWeightedIn0[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewAddWeightedIn1 = AIE.objectFifo.acquire @OF_median_out_5(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemAddWeightedIn1 = AIE.objectFifo.subview.access %subviewAddWeightedIn1[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewAddWeightedOut = AIE.objectFifo.acquire @OF_add_out_5(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemAddWeightedOut = AIE.objectFifo.subview.access %subviewAddWeightedOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %a     = memref.load %rtp05[%c0] : memref<16xi32>
                %b     = memref.load %rtp05[%c1] : memref<16xi32>
                %g     = memref.load %rtp05[%c2] : memref<16xi32>
                %alpha = arith.trunci         %a : i32 to i16
                %beta  = arith.trunci         %b : i32 to i16
                %gamma = arith.trunci         %g : i32 to  i8
                func.call @addWeightedLine(%elemAddWeightedIn0, %elemAddWeightedIn1, %elemAddWeightedOut, %tileWidth, %alpha, %beta, %gamma) : (memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i8) -> ()
                AIE.objectFifo.release @OF_rgba_out_5(Consume, 1)
                AIE.objectFifo.release @OF_median_out_5(Consume, 1)
                AIE.objectFifo.release @OF_add_out_5(Produce, 1)               

                %subviewThresholdIn = AIE.objectFifo.acquire @OF_add_out_5(Consume, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemThreshold0 = AIE.objectFifo.subview.access %subviewThresholdIn[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %subviewThresholdOut = AIE.objectFifo.acquire @outOFL1L2_5(Produce, 1) : !AIE.objectFifoSubview<memref<1920xui8>>
                %elemThresholdOut = AIE.objectFifo.subview.access %subviewThresholdOut[0] : !AIE.objectFifoSubview<memref<1920xui8>> -> memref<1920xui8>
                %tv                  = memref.load %rtp05[%c3] : memref<16xi32>
                %tmb                 = memref.load %rtp05[%c4] : memref<16xi32>
                %mv                  = memref.load %rtp05[%c5] : memref<16xi32>
                %thresholdValue      = arith.trunci        %tv : i32 to i16
                %thresholdModeBinary = arith.trunci       %tmb : i32 to i16
                %maxValue            = arith.trunci        %mv : i32 to i16
                func.call @thresholdLine(%elemThreshold0, %elemThresholdOut, %tileWidth, %thresholdValue, %maxValue, %thresholdModeBinary) : (memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i16) -> ()
                AIE.objectFifo.release @OF_add_out_5(Consume, 1)
                AIE.objectFifo.release @outOFL1L2_5(Produce, 1)
            }

            AIE.end
        } { link_with="combined_rgba2gray_median_addWeighted_threshold.a" } // indicate kernel object name used by this core

        func.func @sequence(%in : memref<2073600xi32>, %out : memref<518400xi32>) {
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i32
            %c2 = arith.constant 2 : i32
            %c4 = arith.constant 4 : i32
            %imageHeight = arith.constant 1080  : i32
            %tileHeight  = arith.constant 10 : i32 // 270  : i32 // 4 horizontal tiles

            %tileWidthRGBA  = arith.constant  1920 : i32  // in 32b words so tileWidth (since there are 4 channels in rgba)
            %tileWidthGray  = arith.constant   480 : i32  // in 32b words so tileWidth/4 (since there is only 1 channels in gray)
            %halfTileWidthRGBA  = arith.constant  810 : i32  // in 32b words so tileWidth (since there are 4 channels in rgba)
            %halfTileWidthGray  = arith.constant   240 : i32  // in 32b words so tileWidth/4 (since there is only 1 channels in gray)
            %tileSizeRGBA = arith.constant   518400 : i32 // width*(height/4)
            %tileSizeGray = arith.constant   129600 : i32 // width*(height/4)/4 (32b transfers) 
            %totalSizeRGBA = arith.constant 2073600 : i32
            %totalSizeGray = arith.constant  518400 : i32

                                                                              //                              noise out         denoise out
            AIEX.ipu.rtp_write(0, 2, 0,     0) { buffer_sym_name = "rtp02" }  // alpha                  16384 = 1 << 14                   0                  
            AIEX.ipu.rtp_write(0, 2, 1, 16384) { buffer_sym_name = "rtp02" }  // beta                 -16384 = -1 << 14     16384 = 1 << 14 
            AIEX.ipu.rtp_write(0, 2, 2,     0) { buffer_sym_name = "rtp02" }  // gamma                                0                   0              
            AIEX.ipu.rtp_write(0, 2, 3,     0) { buffer_sym_name = "rtp02" }  // thresholdValue                      10                   0
            AIEX.ipu.rtp_write(0, 2, 4,     3) { buffer_sym_name = "rtp02" }  // thresholdModeBinary         binary = 0          toZero = 3
            AIEX.ipu.rtp_write(0, 2, 5,   255) { buffer_sym_name = "rtp02" }  // maxValue                           255                 255

            AIEX.ipu.rtp_write(0, 3, 0,     0) { buffer_sym_name = "rtp03" }  // alpha                  16384 = 1 << 14                   0                  
            AIEX.ipu.rtp_write(0, 3, 1, 16384) { buffer_sym_name = "rtp03" }  // beta                 -16384 = -1 << 14     16384 = 1 << 14 
            AIEX.ipu.rtp_write(0, 3, 2,     0) { buffer_sym_name = "rtp03" }  // gamma                                0                   0              
            AIEX.ipu.rtp_write(0, 3, 3,     0) { buffer_sym_name = "rtp03" }  // thresholdValue                      10                   0
            AIEX.ipu.rtp_write(0, 3, 4,     3) { buffer_sym_name = "rtp03" }  // thresholdModeBinary         binary = 0          toZero = 3
            AIEX.ipu.rtp_write(0, 3, 5,   255) { buffer_sym_name = "rtp03" }  // maxValue                           255                 255

            AIEX.ipu.rtp_write(0, 4, 0,     0) { buffer_sym_name = "rtp04" }  // alpha                  16384 = 1 << 14                   0                  
            AIEX.ipu.rtp_write(0, 4, 1, 16384) { buffer_sym_name = "rtp04" }  // beta                 -16384 = -1 << 14     16384 = 1 << 14 
            AIEX.ipu.rtp_write(0, 4, 2,     0) { buffer_sym_name = "rtp04" }  // gamma                                0                   0              
            AIEX.ipu.rtp_write(0, 4, 3,     0) { buffer_sym_name = "rtp04" }  // thresholdValue                      10                   0
            AIEX.ipu.rtp_write(0, 4, 4,     3) { buffer_sym_name = "rtp04" }  // thresholdModeBinary         binary = 0          toZero = 3
            AIEX.ipu.rtp_write(0, 4, 5,   255) { buffer_sym_name = "rtp04" }  // maxValue                           255                 255

            AIEX.ipu.rtp_write(0, 5, 0,     0) { buffer_sym_name = "rtp05" }  // alpha                  16384 = 1 << 14                   0                  
            AIEX.ipu.rtp_write(0, 5, 1, 16384) { buffer_sym_name = "rtp05" }  // beta                 -16384 = -1 << 14     16384 = 1 << 14 
            AIEX.ipu.rtp_write(0, 5, 2,     0) { buffer_sym_name = "rtp05" }  // gamma                                0                   0              
            AIEX.ipu.rtp_write(0, 5, 3,     0) { buffer_sym_name = "rtp05" }  // thresholdValue                      10                   0
            AIEX.ipu.rtp_write(0, 5, 4,     3) { buffer_sym_name = "rtp05" }  // thresholdModeBinary         binary = 0          toZero = 3
            AIEX.ipu.rtp_write(0, 5, 5,   255) { buffer_sym_name = "rtp05" }  // maxValue                           255                 255

            //dma_memcpy_nd ([offset in 32b words][length in 32b words][stride in 32b words])
            //AIEX.ipu.dma_memcpy_nd (%c0, %c0, %in [%c0, %c0, %c0, %c0][%tileHeight, %c4, %c2, %halfTileWidthRGBA][%tileWidthRGBA, %tileSizeRGBA, %halfTileWidthRGBA]) { metadata = "ofIn",  id = 1 : i32 } : (i32, i32, memref<2073600xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
            //AIEX.ipu.dma_memcpy_nd (%c0, %c0, %out[%c0, %c0, %c0, %c0][%tileHeight, %c4, %c2, %halfTileWidthGray][%tileWidthGray, %tileSizeGray, %halfTileWidthGray]) { metadata = "ofOut", id = 0 : i32 } : (i32, i32, memref<518400xi32>,  [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
            AIEX.ipu.dma_memcpy_nd (%c0, %c0, %in [%c0, %c0, %c0, %c0][%c1, %c1, %c1, %totalSizeRGBA][%c0, %c0, %c0]) { metadata = @inOF_L3L2,  id = 1 : i32 } : (i32, i32, memref<2073600xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
            AIEX.ipu.dma_memcpy_nd (%c0, %c0, %out[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %totalSizeGray][%c0, %c0, %c0]) { metadata = @outOFL2L3, id = 0 : i32 } : (i32, i32, memref<518400xi32>,  [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
            AIEX.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
            return
        }
    }
}
