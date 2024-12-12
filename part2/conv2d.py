import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
# from neuronxcc.nki import baremetal




@nki.jit
def tensor_avgpool_kernel(in_tile, out_tile, pool_size):
    """NKI kernel to compute a 2D avg-pool operation

    Args:
            in_tile: an input SBUF tile, of shape C x H x W
            out_tile: an output SBUF tile, of shape C x (H/pool_size) x (W/pool_size)
            pool_size: an integer representing a (square) pool-window size

    Return:
            (None: Result is written to out_tile)
    """

    # Get input/output dimensions
    sz_cin, sz_hin, sz_win = in_tile.shape
    sz_hout = sz_hin // pool_size
    sz_wout = sz_win // pool_size

    # Set relevant sizes
    sz_p = sz_cin
    sz_pool = pool_size

    # Generate pool index patterns (requires two extra dimensions, for the pool window)
    i0, i1, i2, i3, i4 = nl.mgrid[0:sz_p, 0:sz_hin//sz_pool, 0:sz_pool, 0:sz_win//sz_pool, 0:sz_pool]

    # Perform the pooling operation:
    # We use numpy's advanced indexing, in order to extend in_tile to 5D, and then reduce-average two dimension.
    # axis[0] is the index for p_dim, and thus doesn't participate in the reduction operation.
    # axis[1] and axis[2] together index the rows, with axis[2] responsible for inner strides
    # (i.e. inside a pooling window), and axis[1] responsible for the outer strides. As such, we reduce over axis[2].
    # Similarly, axis[3] and axis[4] together index the columns, and we thus reduce over axis[4].

    test_tile = in_tile[
            i0,
            sz_pool*i1+i2,
            sz_pool*i3+i4]

    print(f"\n\ntest_tile.shape: {test_tile.shape}") # (128, 15, 2, 7, 2)


    out_tile[...] = nl.max(in_tile[
            i0,
            sz_pool*i1+i2,
            sz_pool*i3+i4],
            axis=[2,4]) # SBUF:(sz_p, sz_hout, sz_wout)







"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""







@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    filter_vertical_pad = filter_height // 2 # 3px filter -> 1px padding top+bottom

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    d_c_in = nl.tile_size.pmax
    n_tiles_c_in = in_channels // d_c_in

    # Ho tiling dimensions don't factor in pooling
    d_ho = nl.tile_size.gemm_moving_fmax // out_width # e.g. 512 max, 30px out rows = 17 vertical rows
    d_ho = max(min(d_ho, out_height), 1) # Ensure at least 1 row, and not more than the output height
    d_wo = out_width
    n_tiles_ho = (out_height + d_ho - 1) // d_ho # e.g. 30px out rows, 17px vertical rows = 2 vertical tiles
    # n_tiles_wo = out_width // d_wo # It's always 1
    d_k = nl.tile_size.pmax
    n_tiles_k = out_channels // d_k # Number of tiles in K dimension

    # Pre-transpose W to w=(R,S,C,K)
    W = W.reshape((
        n_tiles_k, d_k, 
        n_tiles_c_in, d_c_in, 
        filter_height, filter_width))
        # (K,C,R,S)->(n_k,d_k, n_c,d_c, r,s)
    
    weight_sbuf = nl.ndarray((n_tiles_k, nl.par_dim(d_k), n_tiles_c_in, d_c_in, filter_height, filter_width), dtype = W.dtype, buffer = nl.sbuf) # SBUF:(n_k,d_k, n_c,d_c, r,s)

    weight_copy = nl.ndarray((filter_height, filter_width, n_tiles_k, n_tiles_c_in, nl.par_dim(d_k), d_c_in), dtype=W.dtype, buffer=nl.sbuf) # SBUF:(r,s, n_k,n_c, d_k,d_c)
    w = nl.ndarray(
        (filter_height, filter_width, n_tiles_k, n_tiles_c_in, nl.par_dim(d_c_in), d_k), 
        dtype=W.dtype, buffer=nl.sbuf) # SBUF:(r,s, n_k,n_c, d_c,d_k)

    for k in nl.affine_range(n_tiles_k):
        weight_sbuf[k] = nl.load(W[k]) # Each:(d_k,n_c,d_c,r,s)
    for k in nl.affine_range(n_tiles_k):
        for c in nl.affine_range(n_tiles_c_in):
            for r in nl.affine_range(filter_height):
                for s in nl.affine_range(filter_width):
                    weight_copy[r, s, k, c, :, :] = nl.copy(weight_sbuf[k, :, c, :, r, s], dtype = W.dtype) # SBUF:(r,s,k,c, d_k,d_c)
                    w[r, s, k, c] = nisa.nc_transpose(weight_copy[r, s, k, c]) # SBUF:(r,s,k,c, d_c,d_k)

    # nki.tensor.assert_shape(w, (filter_height, filter_width, n_tiles_k, n_tiles_c_in, d_c_in, d_k)) # SBUF:(r,s, n_k,n_c, d_c,d_k)

    # Process the images in batches
    # print(f"Iterators: b={batch_size}, r={filter_height}, s={filter_width}")
    for b in nl.affine_range(batch_size):
        # TODO: Perform the convolution of X[b] with the weights W and bias b, followed by a maxpool
        # and store the result in X_out[b]

        # Loop over ho rows of pixels
        for ho in nl.affine_range(n_tiles_ho):
            ho_row_start = ho*d_ho # Output pixels are indexed differently from input pixels
            ho_row_end = (ho+1)*d_ho # (Not inclusive, i.e. [start:end] indexing)

            # Loop over k filters
            for k in nl.affine_range(n_tiles_k):

                # Load bias for this group of filters
                cur_bias = nl.load(bias[k*d_k:(k+1)*d_k]) # SBUF:(k,)

                # Setup single (k,howo) tile accumulator in PSUM
                # Hardware quirk: PSUM must always be float32
                cur_out_tile_acc = nl.zeros(
                    (d_k, d_ho*d_wo),
                    dtype=nl.float32, buffer=nl.psum, 
                ) # PSUM:(k,howo)

                # Loop over tiles in C dimension
                for c_in in nl.affine_range(n_tiles_c_in):

                    # Pull out the patch of the input image which surrounds
                    # the output tile's pixels by a buffer enough to contain
                    # the footprint of the filters
                    # (Since we're doing entire image widths at a time,
                    # simply pad the row count above and below)

                    # If we're on output row 0, this is actually input row 1.
                    # Which uses pixels from input rows 0-2.
                    hi_row_start = ho_row_start 
                    # +1 to translate to input index
                    # +1 to account for padding
                    hi_row_end   = ho_row_end + 2*filter_vertical_pad

                    cur_img_input = nl.load(
                        X[b,
                          c_in*d_c_in:(c_in+1)*d_c_in,
                          hi_row_start:hi_row_end, :]) # SBUF:(c,ho^,wo^)

                    # Loop over filter pixels
                    for r in nl.affine_range(filter_height):
                        for s in nl.affine_range(filter_width):

                            # Pull out the specific subarea of the image patch above 
                            # required to compute this filter pixel's contribution to 
                            # this output tile's pixels
                            # (it's simply different offsets of the same size)
                            # Should be (c,ho,wo)
                            h_start = r
                            h_end = r + d_ho
                            w_start = s
                            w_end = s + d_wo

                            cur_img_input_trim = cur_img_input[:, h_start:h_end, w_start:w_end] # (c,ho,wo)
                            cur_img_input_trim = nl.copy(cur_img_input_trim)
                            # nki.tensor.assert_shape(cur_img_input_trim, (d_c_in, d_ho, d_wo))

                            # Reshape for matmul
                            cur_img_input_trim_2d = cur_img_input_trim.reshape(
                                (d_c_in, d_ho*d_wo)) # (c,ho*wo)
                            # nki.tensor.assert_shape(cur_img_input_trim_2d, (d_c_in, d_ho*d_wo))

                            # Setup matrix A^T for cur filter rs (c,k)
                            cur_w = w[r,s,k,c_in] # SBUF:(c,k)
                            # nki.tensor.assert_shape(cur_w, (d_c_in, d_k))

                            # Do the matmul
                            lhsT = cur_w # SBUF:(c,k)
                            rhs = cur_img_input_trim_2d # SBUF:(c,ho*wo)
                            rhs = nl.copy(rhs)
                            cur_out_tile_acc += nl.matmul(lhsT, rhs, transpose_x=True) # PSUM:(k,ho*wo)

                # At this point, we've accumulated every filter pixel and channel for this (k,ho,wo) tile

                # Copy into SBUF, add bias, and reshape
                cur_out_tile_acc_sbuf = nl.copy(cur_out_tile_acc) # SBUF:(k,ho*wo)
                cur_out_tile_acc_sbuf = cur_out_tile_acc_sbuf + cur_bias # SBUF:(k,ho*wo)
                cur_out_tile_acc_sbuf = cur_out_tile_acc_sbuf.reshape(
                    (d_k, d_ho, d_wo)) # SBUF:(k,ho,wo)
                
                # Handle maxpool
                # (inspired by https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/tutorials/average_pool2d.html)
                if pool_size > 1:
                    # Setup advanced indices
                    i_k, i_hp, i_hp_p, i_wp, i_wp_p = nl.mgrid[
                        0:d_k, 
                        0:d_ho//pool_size,
                        0:pool_size,
                        0:d_wo//pool_size,
                        0:pool_size,
                    ]

                    # Reindex to be max-ready (k,hp,p,wp,p)
                    cur_out_reindexed = cur_out_tile_acc_sbuf[
                        i_k,
                        i_hp*pool_size + i_hp_p,
                        i_wp*pool_size + i_wp_p]
                    
                    # Take max
                    cur_out_tile_acc_sbuf = nl.max(
                        cur_out_reindexed, [2,4]
                    ) # SBUF:(k,hp,wp)

                    # Update sizing variables for the .store() below
                    d_ho = d_ho//pool_size
                    d_wo = d_wo//pool_size

                # Store the result in HBM
                nl.store(X_out[
                    b,
                    k*d_k:(k+1)*d_k,
                    ho*d_ho:(ho+1)*d_ho,
                    :],
                    value=cur_out_tile_acc_sbuf) # PSUM->HBM:(k,ho,wo)
                            
    return X_out

