//Macros for the comparisons of float and arrays of floats

pub const F64_EPSILON: f64 = 1e-10;

#[macro_export]
macro_rules! lt {
    ( $v_1:expr, $v_2:expr ) => {
        $v_1 < ($v_2 - crate::macros::F64_EPSILON)
    };
}

#[macro_export]
macro_rules! leq {
    ( $v_1:expr, $v_2:expr ) => {
        $v_1 <= $v_2 + crate::macros::F64_EPSILON
    };
}

#[macro_export]
macro_rules! eq {
    ( $v_1:expr, $v_2:expr ) => {
        !crate::lt!($v_1, $v_2) && !crate::lt!($v_2, $v_1)
    };
}

#[macro_export]
macro_rules! vec_eq {
    ( $uvw_1:expr, $uvw_2:expr ) => {
        crate::eq!($uvw_1[0], $uvw_2[0])
            && crate::eq!($uvw_1[1], $uvw_2[1])
            && crate::eq!($uvw_1[2], $uvw_2[2])
    };
}

#[macro_export]
macro_rules! vec_lt {
    ( $uvw_1:expr, $uvw_2:expr ) => {
        crate::lt!($uvw_1[0], $uvw_2[0])
            || (crate::eq!($uvw_1[0], $uvw_2[0]) && crate::lt!($uvw_1[1], $uvw_2[1]))
            || (crate::eq!($uvw_1[0], $uvw_2[0])
                && crate::eq!($uvw_1[1], $uvw_2[1])
                && crate::lt!($uvw_1[2], $uvw_2[2]))
    };
}

//Macros for some algebraic mainpulations of arrays of floats

#[macro_export]
macro_rules! vec_sp {
    ( $uvw_1:expr, $uvw_2:expr ) => {
        $uvw_1[0] * $uvw_2[0] + $uvw_1[1] * $uvw_2[1] + $uvw_1[2] * $uvw_2[2]
    };
}

#[macro_export]
macro_rules! vec_diff {
    ( $uvw_1:expr, $uvw_2:expr ) => {
        [
            $uvw_1[0] - $uvw_2[0],
            $uvw_1[1] - $uvw_2[1],
            $uvw_1[2] - $uvw_2[2],
        ]
    };
}

// Solve a = l1 d1 + l2 d2 using a least square approach where d1, d2 and a are 3 dimensionnal arrays
#[macro_export]
macro_rules! vec_lsq {
    ( $a:expr, $d1:expr, $d2:expr, $l:expr) => {
        let d11 = crate::vec_sp!($d1, $d1);
        let d22 = crate::vec_sp!($d2, $d2);
        let d12 = crate::vec_sp!($d1, $d2);
        let j = d11 * d22 - d12 * d12;
        let a1 = crate::vec_sp!($a, $d1) / j;
        let a2 = crate::vec_sp!($a, $d2) / j;

        $l = [d22 * a1 - d12 * a2, d11 * a2 - d12 * a1];
    };
}
