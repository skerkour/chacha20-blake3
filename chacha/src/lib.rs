#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]

// aarch64 assumes that NEON instructions are always present
#[cfg(target_arch = "aarch64")]
use crate::chacha_neon::chacha_neon;

#[cfg(target_arch = "aarch64")]
mod chacha_neon;

// import if the target runtime supports the feature
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod chacha_wasm_simd128;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use chacha_wasm_simd128::chacha_wasm_simd128;

// import if runtime CPU features detection is enabled or if the target CPU supports the feature
#[cfg(any(
    all(target_arch = "x86_64", feature = "std"),
    all(target_arch = "x86_64", target_feature = "avx2")
))]
mod chacha_avx2;

#[cfg(any(
    all(target_arch = "x86_64", feature = "std"),
    all(target_arch = "x86_64", target_feature = "avx2")
))]
use chacha_avx2::chacha_avx2;

// import if runtime CPU features detection is enabled or if the target CPU supports the feature
#[cfg(any(
    all(target_arch = "x86_64", feature = "std"),
    all(target_arch = "x86_64", target_feature = "avx512f")
))]
mod chacha_avx512;

#[cfg(any(
    all(target_arch = "x86_64", feature = "std"),
    all(target_arch = "x86_64", target_feature = "avx512f")
))]
use chacha_avx512::chacha_avx512;

#[cfg(feature = "zeroize")]
use zeroize::{Zeroize, ZeroizeOnDrop};

/// The number of 32-bit words that compose ChaCha's state.
const STATE_WORDS: usize = 16;

/// The size of a ChaCha block in bytes which is the size of the state in bytes
const BLOCK_SIZE: usize = 64;

/// The "sigma" constant which is the value of the first row of ChaCha's state.
const CONSTANT: [u32; 4] = [
    0x61707865, // "expa"
    0x3320646e, // "nd 3"
    0x79622d32, // "2-by"
    0x6b206574, // "te k"
];

pub type ChaCha8 = ChaCha<8>;
pub type ChaCha12 = ChaCha<12>;
pub type ChaCha20 = ChaCha<20>;

#[cfg_attr(feature = "zeroize", derive(Zeroize, ZeroizeOnDrop))]
pub struct ChaCha<const ROUNDS: usize> {
    state: [u32; STATE_WORDS],
    /// ChaCha is a stream cipher that works with 64-byte blocks.
    /// It means that consumers of this packages should be able to call `xor_keystream` multiple
    /// times even if there input is not aligned with ChaCha blocks.
    /// Thus calling multiple times `xor_keystream`:
    /// xor_keystream(plaintext[0..3]), xor_keystream(plaintext[3..50]), xor_keystream(plaintext[50..150]);
    /// Should be equal to calling it only once:
    /// xor_keystream(plaintext[0..150]);
    /// For that, we keep the last computed keystream block, as well as an index of where in the keystream
    /// we were after completing the last call.
    /// Then, when calling `xor_keystream` again, we first check if there is sone leftover form the last
    /// keystream.
    /// NOTE: the `last_keystream_block` is valid only if the previous call to `xor_keystream` had
    /// an input.len() % 64 != 0.
    /// Otherwise there is no need to preserve the last keystream block.
    last_keystream_block: [u8; BLOCK_SIZE],
    last_keystream_block_index: usize,
}

impl<const ROUNDS: usize> ChaCha<ROUNDS> {
    pub fn new(key: &[u8; 32], nonce: &[u8; 8]) -> ChaCha<ROUNDS> {
        let mut state = [0u32; STATE_WORDS];

        // copy constant into the first 4 32-bit words
        state[..4].copy_from_slice(&CONSTANT);

        // copy key into state as 4 32-bit little-endian words
        for (state_word, key_chunk) in state[4..12].iter_mut().zip(key.chunks_exact(4)) {
            *state_word = u32::from_le_bytes(key_chunk.try_into().unwrap());
        }

        // counter is already initialized to 0
        // state[12] = 0;
        // state[13] = 0;

        // copy nonce into state as 2 32-bit little-endian words
        state[14] = u32::from_le_bytes(nonce[0..4].try_into().unwrap());
        state[15] = u32::from_le_bytes(nonce[4..8].try_into().unwrap());

        return ChaCha {
            state,
            last_keystream_block: [0u8; BLOCK_SIZE],
            last_keystream_block_index: 0,
        };
    }

    /// XOR `plaintext` with the ChaCha keystream.
    pub fn xor_keystream(&mut self, mut plaintext: &mut [u8]) {
        if plaintext.len() == 0 {
            return;
        }

        // first, consume the keystream leftover, if any
        if self.last_keystream_block_index != 0 {
            let remaining_keystream = &self.last_keystream_block[self.last_keystream_block_index..];

            plaintext
                .iter_mut()
                .zip(remaining_keystream)
                .for_each(|(plaintext, keystream)| *plaintext ^= *keystream);

            if plaintext.len() > remaining_keystream.len() {
                plaintext = &mut plaintext[remaining_keystream.len()..];
            } else if plaintext.len() < remaining_keystream.len() {
                self.last_keystream_block_index += plaintext.len();
                return;
            } else {
                // plaintext.len() == remaining_keystream.len()
                self.last_keystream_block_index = 0;
                return;
            }
        }
        self.last_keystream_block_index = plaintext.len() % BLOCK_SIZE;

        // aarch64 assumes that NEON is always available
        #[cfg(target_arch = "aarch64")]
        if plaintext.len() >= 128 {
            chacha_neon::<ROUNDS>(&mut self.state, plaintext, &mut self.last_keystream_block);
            return;
        }

        // wasm32 only supports compile-time features detection
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        if plaintext.len() >= 128 {
            chacha_wasm_simd128::<ROUNDS>(&mut self.state, plaintext, &mut self.last_keystream_block);
            return;
        }

        // runtime detection of CPU features for x86 and x86_64 when the "std" feature is enabled
        #[cfg(feature = "std")]
        {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") && plaintext.len() >= 128 {
                // SAFETY: we just verified AVX-512 is available via runtime detection.
                unsafe {
                    chacha_avx512::<ROUNDS>(&mut self.state, plaintext, &mut self.last_keystream_block);
                }
                return;
            }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if is_x86_feature_detected!("avx2") && plaintext.len() >= 128 {
                // SAFETY: we just verified AVX2 is available via runtime detection.
                unsafe {
                    chacha_avx2::<ROUNDS>(&mut self.state, plaintext, &mut self.last_keystream_block);
                }
                return;
            }
        }

        // compile-time CPU features detection for x86 and x86_64 when the "std" feature is not enabled
        #[cfg(not(feature = "std"))]
        {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            if plaintext.len() >= 128 {
                // SAFETY: target_feature = "avx512f" guarantees AVX-512 at compile time.
                unsafe {
                    chacha_avx512::<ROUNDS>(&mut self.state, plaintext, &mut self.last_keystream_block);
                }
                return;
            }

            #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
            if plaintext.len() >= 128 {
                // SAFETY: target_feature = "avx2" guarantees AVX2 at compile time.
                unsafe {
                    chacha_avx2::<ROUNDS>(&mut self.state, plaintext, &mut self.last_keystream_block);
                }
                return;
            }
        }

        // fallback for when SIMD acceleration is not available
        chacha_generic::<ROUNDS>(&mut self.state, &mut self.last_keystream_block, plaintext);
    }

    /// Set the ChaCha counter (words 12 and 13). It can be used to move forward and backward in the
    /// keystream.
    pub fn set_counter(&mut self, counter: u64) {
        inject_counter_into_state(&mut self.state, counter);
        // setting the counter "realigns" the keystream to the beginning of a block.
        self.last_keystream_block_index = 0;
    }
}

#[inline]
fn chacha_generic<const ROUNDS: usize>(
    mut state: &mut [u32; STATE_WORDS],
    last_keystream_block: &mut [u8; BLOCK_SIZE],
    plaintext: &mut [u8],
) {
    let mut keystream = [0u8; BLOCK_SIZE];
    let keystream_ptr = keystream.as_mut_ptr();
    let mut counter = extract_counter_from_state(state);

    // process the input by blocks of 64 bytes
    for plaintext_block in plaintext.chunks_mut(BLOCK_SIZE) {
        inject_counter_into_state(&mut state, counter);

        // prepare temporary (working) state
        let mut tmp_state = *state;

        // perform the ROUNDS / 2 double rounds e.g. 10 double rounds for ChaCha20
        for _ in 0..(ROUNDS / 2) {
            // column rounds
            quarter_round(&mut tmp_state, 0, 4, 8, 12);
            quarter_round(&mut tmp_state, 1, 5, 9, 13);
            quarter_round(&mut tmp_state, 2, 6, 10, 14);
            quarter_round(&mut tmp_state, 3, 7, 11, 15);

            // diagonal rounds
            quarter_round(&mut tmp_state, 0, 5, 10, 15);
            quarter_round(&mut tmp_state, 1, 6, 11, 12);
            quarter_round(&mut tmp_state, 2, 7, 8, 13);
            quarter_round(&mut tmp_state, 3, 4, 9, 14);
        }

        // add initial state to tmp_state to generate the keystream and "serialize" it to little endian
        // for (tmp_word, state_word) in tmp_state.iter_mut().zip(state.iter()) {
        //     *tmp_word = tmp_word.wrapping_add(*state_word).to_le();
        // }
        for word_index in 0..STATE_WORDS {
            // first we add the initial state to the working state to get the keystream
            tmp_state[word_index] = tmp_state[word_index].wrapping_add(state[word_index]);

            // then we serialize the keystream
            unsafe {
                core::ptr::copy_nonoverlapping(
                    tmp_state[word_index].to_le_bytes().as_ptr(),
                    keystream_ptr.add(word_index * 4),
                    4,
                );
            }
        }

        // XOR plaintext with keystream
        plaintext_block
            .iter_mut()
            .zip(keystream)
            .for_each(|(plaintext, keystream)| *plaintext ^= keystream);

        counter = counter.wrapping_add(1);
    }

    inject_counter_into_state(state, counter);

    if plaintext.len() % BLOCK_SIZE != 0 {
        last_keystream_block.copy_from_slice(&keystream);
    }
}

#[inline(always)]
const fn quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    // a += b; d ^= a; d <<<= 16
    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(16);

    // c += d; b ^= c; b <<<= 12
    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(12);

    // a += b; d ^= a; d <<<= 8
    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(8);

    // c += d; b ^= c; b <<<= 7
    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(7);
}

#[inline(always)]
fn extract_counter_from_state(state: &[u32; STATE_WORDS]) -> u64 {
    return ((state[13] as u64) << 32) | (state[12] as u64);
}

#[inline(always)]
fn inject_counter_into_state(state: &mut [u32; STATE_WORDS], counter: u64) {
    state[12] = counter as u32;
    state[13] = (counter >> 32) as u32;
}

#[cfg(test)]
mod test {
    use crate::{ChaCha8, ChaCha12, ChaCha20};

    struct Test {
        key: [u8; 32],
        nonce: [u8; 8],
        initial_counter: u64,
        plaintext: Vec<u8>,
        expected_ciphertext: Vec<u8>,
    }

    #[test]
    fn chacha20_test_vectors() {
        let tests = vec![
            // https://www.rfc-editor.org/rfc/rfc8439#section-2.4.2
            Test {
                key: hex::decode("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
                    .unwrap()
                    .try_into()
                    .unwrap(),
                nonce: hex::decode("0000004a00000000").unwrap().try_into().unwrap(),
                initial_counter: 1,
                plaintext: hex::decode(
                    "4c616469657320616e642047656e746c\
656d656e206f662074686520636c6173\
73206f66202739393a20496620492063\
6f756c64206f6666657220796f75206f\
6e6c79206f6e652074697020666f7220\
746865206675747572652c2073756e73\
637265656e20776f756c642062652069\
742e",
                )
                .unwrap(),
                expected_ciphertext: hex::decode(
                    "6e2e359a2568f98041ba0728dd0d6981\
e97e7aec1d4360c20a27afccfd9fae0b\
f91b65c5524733ab8f593dabcd62b357\
1639d624e65152ab8f530c359f0861d8\
07ca0dbf500d6a6156a38e088a22b65e\
52bc514d16ccf806818ce91ab7793736\
5af90bbf74a35be6b40b8eedf2785e42\
874d",
                )
                .unwrap(),
            },
            // https://www.rfc-editor.org/rfc/rfc8439#appendix-A.2 Test vector #1
            Test {
                key: [0u8; 32],
                nonce: [0u8; 8],
                initial_counter: 0,
                plaintext: [0u8; 64].to_vec(),
                expected_ciphertext: hex::decode(
                    "76b8e0ada0f13d90405d6ae55386bd28\
bdd219b8a08ded1aa836efcc8b770dc7\
da41597c5157488d7724e03fb8d84a37\
6a43b8f41518a11cc387b669b2ee6586",
                )
                .unwrap(),
            },
            // https://www.rfc-editor.org/rfc/rfc8439#appendix-A.2 Test Vector #2
            Test {
                key: hex::decode("0000000000000000000000000000000000000000000000000000000000000001")
                    .unwrap()
                    .try_into()
                    .unwrap(),
                nonce: hex::decode("0000000000000002").unwrap().try_into().unwrap(),
                initial_counter: 1,
                plaintext: hex::decode(
                    "416e79207375626d697373696f6e2074\
6f20746865204945544620696e74656e\
6465642062792074686520436f6e7472\
696275746f7220666f72207075626c69\
636174696f6e20617320616c6c206f72\
2070617274206f6620616e2049455446\
20496e7465726e65742d447261667420\
6f722052464320616e6420616e792073\
746174656d656e74206d616465207769\
7468696e2074686520636f6e74657874\
206f6620616e20494554462061637469\
7669747920697320636f6e7369646572\
656420616e20224945544620436f6e74\
7269627574696f6e222e205375636820\
73746174656d656e747320696e636c75\
6465206f72616c2073746174656d656e\
747320696e2049455446207365737369\
6f6e732c2061732077656c6c20617320\
7772697474656e20616e6420656c6563\
74726f6e696320636f6d6d756e696361\
74696f6e73206d61646520617420616e\
792074696d65206f7220706c6163652c\
20776869636820617265206164647265\
7373656420746f",
                )
                .unwrap(),
                expected_ciphertext: hex::decode(
                    "a3fbf07df3fa2fde4f376ca23e827370\
41605d9f4f4f57bd8cff2c1d4b7955ec\
2a97948bd3722915c8f3d337f7d37005\
0e9e96d647b7c39f56e031ca5eb6250d\
4042e02785ececfa4b4bb5e8ead0440e\
20b6e8db09d881a7c6132f420e527950\
42bdfa7773d8a9051447b3291ce1411c\
680465552aa6c405b7764d5e87bea85a\
d00f8449ed8f72d0d662ab052691ca66\
424bc86d2df80ea41f43abf937d3259d\
c4b2d0dfb48a6c9139ddd7f76966e928\
e635553ba76c5c879d7b35d49eb2e62b\
0871cdac638939e25e8a1e0ef9d5280f\
a8ca328b351c3c765989cbcf3daa8b6c\
cc3aaf9f3979c92b3720fc88dc95ed84\
a1be059c6499b9fda236e7e818b04b0b\
c39c1e876b193bfe5569753f88128cc0\
8aaa9b63d1a16f80ef2554d7189c411f\
5869ca52c5b83fa36ff216b9c1d30062\
bebcfd2dc5bce0911934fda79a86f6e6\
98ced759c3ff9b6477338f3da4f9cd85\
14ea9982ccafb341b2384dd902f3d1ab\
7ac61dd29c6f21ba5b862f3730e37cfd\
c4fd806c22f221",
                )
                .unwrap(),
            },
            // https://www.rfc-editor.org/rfc/rfc8439#appendix-A.2 Test Vector #3
            Test {
                key: hex::decode("1c9240a5eb55d38af333888604f6b5f0473917c1402b80099dca5cbc207075c0")
                    .unwrap()
                    .try_into()
                    .unwrap(),
                nonce: hex::decode("0000000000000002").unwrap().try_into().unwrap(),
                initial_counter: 42,
                plaintext: hex::decode(
                    "2754776173206272696c6c69672c2061\
6e642074686520736c6974687920746f\
7665730a446964206779726520616e64\
2067696d626c6520696e207468652077\
6162653a0a416c6c206d696d73792077\
6572652074686520626f726f676f7665\
732c0a416e6420746865206d6f6d6520\
7261746873206f757467726162652e",
                )
                .unwrap(),
                expected_ciphertext: hex::decode(
                    "62e6347f95ed87a45ffae7426f27a1df\
5fb69110044c0d73118effa95b01e5cf\
166d3df2d721caf9b21e5fb14c616871\
fd84c54f9d65b283196c7fe4f60553eb\
f39c6402c42234e32a356b3e764312a6\
1a5532055716ead6962568f87d3f3f77\
04c6a8d1bcd1bf4d50d6154b6da731b1\
87b58dfd728afa36757a797ac188d1",
                )
                .unwrap(),
            },
        ];

        for (i, test) in tests.into_iter().enumerate() {
            let mut cipher = ChaCha20::new(&test.key, &test.nonce);
            cipher.set_counter(test.initial_counter);

            let mut plaintext = test.plaintext.clone();
            cipher.xor_keystream(&mut plaintext);

            assert_eq!(
                plaintext,
                test.expected_ciphertext,
                "test [{i}] failed
Got ciphertext: {}
Expected ciphertext: {}",
                hex::encode(&plaintext),
                hex::encode(&test.expected_ciphertext),
            );

            let mut cipher = ChaCha20::new(&test.key, &test.nonce);
            cipher.set_counter(test.initial_counter);
            cipher.xor_keystream(&mut plaintext);

            assert_eq!(
                plaintext,
                test.plaintext,
                "test [{i}] failed. Initial plaintext != decrypt(encrypt(plaintext))
Got: {}
Expected: {}",
                hex::encode(&plaintext),
                hex::encode(&test.plaintext),
            );

            // ensure that the encryption is correct even for plaintexts that are not % 64 (block size)
            // thus:
            // cipher.xor_keystream(plaintext[0..10])
            // cipher.xor_keystream(plaintext[10..30])
            // cipher.xor_keystream(plaintext[30..5])
            // should be equal to:
            // cipher.xor_keystream(plaintext[0..35])

            let mut cipher = ChaCha20::new(&test.key, &test.nonce);
            cipher.xor_keystream(&mut plaintext);
            for n in 0..10 {
                let mut partial_plaintext: Vec<u8> = test.plaintext.clone();

                let mut cipher = ChaCha20::new(&test.key, &test.nonce);
                cipher.xor_keystream(&mut partial_plaintext[..n]);
                cipher.xor_keystream(&mut partial_plaintext[n..]);

                assert_eq!(
                    plaintext,
                    partial_plaintext,
                    "test [{i}] failed. partial encryption is not valid for n = {n}
            Got: {}
            Expected: {}",
                    hex::encode(&partial_plaintext),
                    hex::encode(&plaintext),
                )
            }
        }
    }

    #[test]
    fn chacha12_case_1() {
        let nonce: &[u8; 8] = &[0xdb, 0x4b, 0x4a, 0x41, 0xd8, 0xdf, 0x18, 0xaa];
        let key: &[u8; 32] = &[
            0x27, 0xfc, 0x12, 0x0b, 0x01, 0x3b, 0x82, 0x9f, 0x1f, 0xae, 0xef, 0xd1, 0xab, 0x41, 0x7e, 0x86, 0x62, 0xf4,
            0x3e, 0x0d, 0x73, 0xf9, 0x8d, 0xe8, 0x66, 0xe3, 0x46, 0x35, 0x31, 0x80, 0xfd, 0xb7,
        ];

        let mut buffer = [0u8; 100];
        ChaCha12::new(key, nonce).xor_keystream(&mut buffer);

        assert_eq!(
            buffer,
            [
                0x5f, 0x3c, 0x8c, 0x19, 0x0a, 0x78, 0xab, 0x7f, 0xe8, 0x08, 0xca, 0xe9, 0xcb, 0xcb, 0x0a, 0x98, 0x37,
                0xc8, 0x93, 0x49, 0x2d, 0x96, 0x3a, 0x1c, 0x2e, 0xda, 0x6c, 0x15, 0x58, 0xb0, 0x2c, 0x83, 0xfc, 0x02,
                0xa4, 0x4c, 0xbb, 0xb7, 0xe6, 0x20, 0x4d, 0x51, 0xd1, 0xc2, 0x43, 0x0e, 0x9c, 0x0b, 0x58, 0xf2, 0x93,
                0x7b, 0xf5, 0x93, 0x84, 0x0c, 0x85, 0x0b, 0xda, 0x90, 0x51, 0xa1, 0xf0, 0x51, 0xdd, 0xf0, 0x9d, 0x2a,
                0x03, 0xeb, 0xf0, 0x9f, 0x01, 0xbd, 0xba, 0x9d, 0xa0, 0xb6, 0xda, 0x79, 0x1b, 0x2e, 0x64, 0x56, 0x41,
                0x04, 0x7d, 0x11, 0xeb, 0xf8, 0x50, 0x87, 0xd4, 0xde, 0x5c, 0x01, 0x5f, 0xdd, 0xd0, 0x44,
            ]
        );
    }

    #[test]
    fn chacha8_case_1() {
        let key = &[
            0x64, 0x1a, 0xea, 0xeb, 0x08, 0x03, 0x6b, 0x61, 0x7a, 0x42, 0xcf, 0x14, 0xe8, 0xc5, 0xd2, 0xd1, 0x15, 0xf8,
            0xd7, 0xcb, 0x6e, 0xa5, 0xe2, 0x8b, 0x9b, 0xfa, 0xf8, 0x3e, 0x03, 0x84, 0x26, 0xa7,
        ];
        let nonce = &[0xa1, 0x4a, 0x11, 0x68, 0x27, 0x1d, 0x45, 0x9b];

        let mut buffer = [0u8; 100];
        ChaCha8::new(key, nonce).xor_keystream(&mut buffer);

        assert_eq!(
            buffer,
            [
                0x17, 0x21, 0xc0, 0x44, 0xa8, 0xa6, 0x45, 0x35, 0x22, 0xdd, 0xdb, 0x31, 0x43, 0xd0, 0xbe, 0x35, 0x12,
                0x63, 0x3c, 0xa3, 0xc7, 0x9b, 0xf8, 0xcc, 0xc3, 0x59, 0x4c, 0xb2, 0xc2, 0xf3, 0x10, 0xf7, 0xbd, 0x54,
                0x4f, 0x55, 0xce, 0x0d, 0xb3, 0x81, 0x23, 0x41, 0x2d, 0x6c, 0x45, 0x20, 0x7d, 0x5c, 0xf9, 0xaf, 0x0c,
                0x6c, 0x68, 0x0c, 0xce, 0x1f, 0x7e, 0x43, 0x38, 0x8d, 0x1b, 0x03, 0x46, 0xb7, 0x13, 0x3c, 0x59, 0xfd,
                0x6a, 0xf4, 0xa5, 0xa5, 0x68, 0xaa, 0x33, 0x4c, 0xcd, 0xc3, 0x8a, 0xf5, 0xac, 0xe2, 0x01, 0xdf, 0x84,
                0xd0, 0xa3, 0xca, 0x22, 0x54, 0x94, 0xca, 0x62, 0x09, 0x34, 0x5f, 0xcf, 0x30, 0x13, 0x2e,
            ]
        );
    }
}
