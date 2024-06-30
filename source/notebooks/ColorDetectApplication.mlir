module  {
   AIE.device(ipu){

   %tile00 = AIE.tile(0, 0)
   %tile01 = AIE.tile(0, 1)
   %tile02 = AIE.tile(0, 2)
   %rtp_0_2 = AIE.buffer(%tile02) { sym_name = "rtp_0_2" } : memref<4xi32>
   %tile03 = AIE.tile(0, 3)
   %rtp_0_3 = AIE.buffer(%tile03) { sym_name = "rtp_0_3" } : memref<3xi32>
   %tile04 = AIE.tile(0, 4)
   %rtp_0_4 = AIE.buffer(%tile04) { sym_name = "rtp_0_4" } : memref<5xi32>
   %tile05 = AIE.tile(0, 5)
   %rtp_0_5 = AIE.buffer(%tile05) { sym_name = "rtp_0_5" } : memref<3xi32>
   AIE.objectFifo @itbuffer_0___ITout___mtbuffer_0___MTin(%tile00, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<1280xi32>>
   AIE.objectFifo @rgba2hue_0___out_buffer___in_range_0___in_buffer(%tile05, {%tile04}, 2 : i32) : !AIE.objectFifo<memref<320xi32>>
   AIE.objectFifo @in_range_0___out_buffer___gray2rgba_0___in_buffer(%tile04, {%tile03}, 2 : i32) : !AIE.objectFifo<memref<320xi32>>
   AIE.objectFifo @gray2rgba_0___out_buffer___bitwiseand_0___in_buffer1(%tile03, {%tile02}, 2 : i32) : !AIE.objectFifo<memref<1280xi32>>
   AIE.objectFifo @bitwiseand_0___out_buffer___itbuffer_1___ITin(%tile02, {%tile00}, 2 : i32) : !AIE.objectFifo<memref<1280xi32>>
   AIE.objectFifo @mtbuffer_0__MTout(%tile01, {%tile05, %tile02}, [2,2,2]) : !AIE.objectFifo<memref<1280xi32>>

   AIE.objectFifo.link [@itbuffer_0___ITout___mtbuffer_0___MTin ] -> [@mtbuffer_0__MTout] ()

   func.func private @in_range(memref<320xi32>, memref<320xi32>, i32, i32, i32) -> ()
   func.func private @rgba2hue(memref<1280xi32>, memref<320xi32>, i32) -> ()
   func.func private @gray2rgba(memref<320xi32>, memref<1280xi32>, i32) -> ()
   func.func private @bitwiseand(memref<1280xi32>, memref<1280xi32>, memref<1280xi32>, i32) -> ()

   AIE.core(%tile02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      %c_rtpidx_3 = arith.constant 3 : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview4 = AIE.objectFifo.acquire @gray2rgba_0___out_buffer___bitwiseand_0___in_buffer1(Consume, 1) : !AIE.objectFifoSubview<memref<1280xi32>>
         %elem4 = AIE.objectFifo.subview.access %subview4[0] : !AIE.objectFifoSubview<memref<1280xi32>> -> memref<1280xi32>
         %subview6 = AIE.objectFifo.acquire @mtbuffer_0__MTout(Consume, 1) : !AIE.objectFifoSubview<memref<1280xi32>>
         %elem6 = AIE.objectFifo.subview.access %subview6[0] : !AIE.objectFifoSubview<memref<1280xi32>> -> memref<1280xi32>
         %subview5 = AIE.objectFifo.acquire @bitwiseand_0___out_buffer___itbuffer_1___ITin(Produce, 1) : !AIE.objectFifoSubview<memref<1280xi32>>
         %elem5 = AIE.objectFifo.subview.access %subview5[0] : !AIE.objectFifoSubview<memref<1280xi32>> -> memref<1280xi32>

         %nbytes = memref.load %rtp_0_2[%c_rtpidx_3] : memref<4xi32>
         func.call @bitwiseand(%elem4, %elem6, %elem5, %nbytes) : (memref<1280xi32>, memref<1280xi32>, memref<1280xi32>, i32) -> ()

         AIE.objectFifo.release @gray2rgba_0___out_buffer___bitwiseand_0___in_buffer1(Consume, 1)
         AIE.objectFifo.release @mtbuffer_0__MTout(Consume, 1)
         AIE.objectFifo.release @bitwiseand_0___out_buffer___itbuffer_1___ITin(Produce, 1)
      }
      AIE.end
   } { link_with="bitwiseand.o" }

   AIE.core(%tile03) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      %c_rtpidx_2 = arith.constant 2 : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview3 = AIE.objectFifo.acquire @in_range_0___out_buffer___gray2rgba_0___in_buffer(Consume, 1) : !AIE.objectFifoSubview<memref<320xi32>>
         %elem3 = AIE.objectFifo.subview.access %subview3[0] : !AIE.objectFifoSubview<memref<320xi32>> -> memref<320xi32>
         %subview4 = AIE.objectFifo.acquire @gray2rgba_0___out_buffer___bitwiseand_0___in_buffer1(Produce, 1) : !AIE.objectFifoSubview<memref<1280xi32>>
         %elem4 = AIE.objectFifo.subview.access %subview4[0] : !AIE.objectFifoSubview<memref<1280xi32>> -> memref<1280xi32>

         %nbytes = memref.load %rtp_0_3[%c_rtpidx_2] : memref<3xi32>
         func.call @gray2rgba(%elem3, %elem4, %nbytes) : (memref<320xi32>, memref<1280xi32>, i32) -> ()

         AIE.objectFifo.release @in_range_0___out_buffer___gray2rgba_0___in_buffer(Consume, 1)
         AIE.objectFifo.release @gray2rgba_0___out_buffer___bitwiseand_0___in_buffer1(Produce, 1)
      }
      AIE.end
   } { link_with="gray2rgba.o" }

   AIE.core(%tile04) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      %c_rtpidx_2 = arith.constant 2 : index
      %c_rtpidx_3 = arith.constant 3 : index
      %c_rtpidx_4 = arith.constant 4 : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview2 = AIE.objectFifo.acquire @rgba2hue_0___out_buffer___in_range_0___in_buffer(Consume, 1) : !AIE.objectFifoSubview<memref<320xi32>>
         %elem2 = AIE.objectFifo.subview.access %subview2[0] : !AIE.objectFifoSubview<memref<320xi32>> -> memref<320xi32>
         %subview3 = AIE.objectFifo.acquire @in_range_0___out_buffer___gray2rgba_0___in_buffer(Produce, 1) : !AIE.objectFifoSubview<memref<320xi32>>
         %elem3 = AIE.objectFifo.subview.access %subview3[0] : !AIE.objectFifoSubview<memref<320xi32>> -> memref<320xi32>

         %nbytes = memref.load %rtp_0_4[%c_rtpidx_2] : memref<5xi32>
         %range_low = memref.load %rtp_0_4[%c_rtpidx_3] : memref<5xi32>
         %range_high = memref.load %rtp_0_4[%c_rtpidx_4] : memref<5xi32>
         func.call @in_range(%elem2, %elem3, %nbytes, %range_low, %range_high) : (memref<320xi32>, memref<320xi32>, i32, i32, i32) -> ()

         AIE.objectFifo.release @rgba2hue_0___out_buffer___in_range_0___in_buffer(Consume, 1)
         AIE.objectFifo.release @in_range_0___out_buffer___gray2rgba_0___in_buffer(Produce, 1)
      }
      AIE.end
   } { link_with="in_range.o" }

   AIE.core(%tile05) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      %c_rtpidx_2 = arith.constant 2 : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview6 = AIE.objectFifo.acquire @mtbuffer_0__MTout(Consume, 1) : !AIE.objectFifoSubview<memref<1280xi32>>
         %elem6 = AIE.objectFifo.subview.access %subview6[0] : !AIE.objectFifoSubview<memref<1280xi32>> -> memref<1280xi32>
         %subview2 = AIE.objectFifo.acquire @rgba2hue_0___out_buffer___in_range_0___in_buffer(Produce, 1) : !AIE.objectFifoSubview<memref<320xi32>>
         %elem2 = AIE.objectFifo.subview.access %subview2[0] : !AIE.objectFifoSubview<memref<320xi32>> -> memref<320xi32>

         %nbytes = memref.load %rtp_0_5[%c_rtpidx_2] : memref<3xi32>
         func.call @rgba2hue(%elem6, %elem2, %nbytes) : (memref<1280xi32>, memref<320xi32>, i32) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout(Consume, 1)
         AIE.objectFifo.release @rgba2hue_0___out_buffer___in_range_0___in_buffer(Produce, 1)
      }
      AIE.end
   } { link_with="rgba2hue.o" }

func.func @sequence(%itbuffer_0 : memref<720x1280xi32>,%itbuffer_1 : memref<720x1280xi32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c5120 = arith.constant 5120 : i32
    %c720 = arith.constant 720 : i32
    %c180 = arith.constant 180 : i32
    %c320 = arith.constant 320 : i32
    %c1280 = arith.constant 1280 : i32

    AIEX.ipu.rtp_write(0, 2, 3, 3) { buffer_sym_name = "rtp_0_2" }
    AIEX.ipu.rtp_write(0, 3, 2, 2) { buffer_sym_name = "rtp_0_3" }
    AIEX.ipu.rtp_write(0, 4, 2, 2) { buffer_sym_name = "rtp_0_4" }
    AIEX.ipu.rtp_write(0, 4, 3, 3) { buffer_sym_name = "rtp_0_4" }
    AIEX.ipu.rtp_write(0, 4, 4, 4) { buffer_sym_name = "rtp_0_4" }
    AIEX.ipu.rtp_write(0, 5, 2, 2) { buffer_sym_name = "rtp_0_5" }

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_1[%c0, %c0, %c0, %c0][%c1, %c1, %c720, %c1280][%c0, %c0, %c0]){ metadata= @bitwiseand_0___out_buffer___itbuffer_1___ITin, id = 1 : i32 } :(i32, i32, memref<720x1280xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_0[%c0, %c0, %c0, %c0][%c1, %c1, %c720, %c1280][%c0, %c0, %c0]){ metadata= @itbuffer_0___ITout___mtbuffer_0___MTin, id = 0 : i32 } :(i32, i32, memref<720x1280xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.sync {column = 0 : i32, row = 0 : i32, direction = 0 : i32, channel = 0 : i32, column_num = 1 : i32, row_num = 1 : i32 }
    return
}
 }
}
