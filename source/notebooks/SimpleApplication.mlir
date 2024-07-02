module  {
   AIE.device(ipu){

   %tile00 = AIE.tile(0, 0)
   %tile05 = AIE.tile(0, 5)
   %rtp_0_5 = AIE.buffer(%tile05) { sym_name = "rtp_0_5" } : memref<6xi32>
   AIE.objectFifo @itbuffer_0___ITout___rgba_rtp_thresh_0___in_buffer(%tile00, {%tile05}, 2 : i32) : !AIE.objectFifo<memref<1280xi32>>
   AIE.objectFifo @rgba_rtp_thresh_0___out_buffer___itbuffer_1___ITin(%tile05, {%tile00}, 2 : i32) : !AIE.objectFifo<memref<1280xi32>>


   func.func private @rgba_rtp_thresh(memref<1280xi32>, memref<1280xi32>, i32, i32, i32, i32) -> ()

   AIE.core(%tile05) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      %c_rtpidx_2 = arith.constant 2 : index
      %c_rtpidx_3 = arith.constant 3 : index
      %c_rtpidx_4 = arith.constant 4 : index
      %c_rtpidx_5 = arith.constant 5 : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview1 = AIE.objectFifo.acquire @itbuffer_0___ITout___rgba_rtp_thresh_0___in_buffer(Consume, 1) : !AIE.objectFifoSubview<memref<1280xi32>>
         %elem1 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<1280xi32>> -> memref<1280xi32>
         %subview2 = AIE.objectFifo.acquire @rgba_rtp_thresh_0___out_buffer___itbuffer_1___ITin(Produce, 1) : !AIE.objectFifoSubview<memref<1280xi32>>
         %elem2 = AIE.objectFifo.subview.access %subview2[0] : !AIE.objectFifoSubview<memref<1280xi32>> -> memref<1280xi32>

         %nbytes = memref.load %rtp_0_5[%c_rtpidx_2] : memref<6xi32>
         %r_thresh = memref.load %rtp_0_5[%c_rtpidx_3] : memref<6xi32>
         %g_thresh = memref.load %rtp_0_5[%c_rtpidx_4] : memref<6xi32>
         %b_thresh = memref.load %rtp_0_5[%c_rtpidx_5] : memref<6xi32>
         func.call @rgba_rtp_thresh(%elem1, %elem2, %nbytes, %r_thresh, %g_thresh, %b_thresh) : (memref<1280xi32>, memref<1280xi32>, i32, i32, i32, i32) -> ()

         AIE.objectFifo.release @itbuffer_0___ITout___rgba_rtp_thresh_0___in_buffer(Consume, 1)
         AIE.objectFifo.release @rgba_rtp_thresh_0___out_buffer___itbuffer_1___ITin(Produce, 1)
      }
      AIE.end
   } { link_with="rgba_rtp_thresh.o" }

func.func @sequence(%itbuffer_0 : memref<720x1280xi32>,%itbuffer_1 : memref<720x1280xi32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c5120 = arith.constant 5120 : i32
    %c720 = arith.constant 720 : i32
    %c180 = arith.constant 180 : i32
    %c320 = arith.constant 320 : i32
    %c1280 = arith.constant 1280 : i32

    AIEX.ipu.rtp_write(0, 5, 2, 2) { buffer_sym_name = "rtp_0_5" }
    AIEX.ipu.rtp_write(0, 5, 3, 3) { buffer_sym_name = "rtp_0_5" }
    AIEX.ipu.rtp_write(0, 5, 4, 4) { buffer_sym_name = "rtp_0_5" }
    AIEX.ipu.rtp_write(0, 5, 5, 5) { buffer_sym_name = "rtp_0_5" }

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_1[%c0, %c0, %c0, %c0][%c1, %c1, %c720, %c1280][%c0, %c0, %c0]){ metadata= @rgba_rtp_thresh_0___out_buffer___itbuffer_1___ITin, id = 1 : i32 } :(i32, i32, memref<720x1280xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_0[%c0, %c0, %c0, %c0][%c1, %c1, %c720, %c1280][%c0, %c0, %c0]){ metadata= @itbuffer_0___ITout___rgba_rtp_thresh_0___in_buffer, id = 0 : i32 } :(i32, i32, memref<720x1280xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.sync {column = 0 : i32, row = 0 : i32, direction = 0 : i32, channel = 0 : i32, column_num = 1 : i32, row_num = 1 : i32 }
    return
}
 }
}
