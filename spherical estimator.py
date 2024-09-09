import numpy as np
from scipy.spatial.transform import Rotation as R

def sampson_error(ri_data, ti_data, rj_data, tj_data, u_data, v_data):
    Ri = R.from_rotvec(ri_data).as_matrix()
    ti = np.array(ti_data)

    Rj = R.from_rotvec(rj_data).as_matrix()
    tj = np.array(tj_data)

    R = Rj @ Ri.T
    t = Rj @ (-Ri.T @ ti) + tj

    s = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    E = s @ R

    u = np.array(u_data)
    v = np.array(v_data)

    line = E @ (u / u[2])
    d = np.dot(v, line)
    
    residuals = d / np.sqrt(line[0]**2 + line[1]**2)
    
    return residuals

def evaluate_model_on_point(E, i, correspondences):
    if i >= len(correspondences):
        print(f"error: {i} / {len(correspondences)}")
    ray_pair = correspondences[i]
    u = ray_pair[0]
    v = ray_pair[1]
    line = E @ (u / u[2])
    d = np.dot(v, line)
    
    return (d * d) / (line[0]**2 + line[1]**2)

def minimal_solver(sample, correspondences, min_sample_size):
    N = len(sample)
    if N < min_sample_size:
        print(f"bad sample size: {N}")
        return 0

    A = np.zeros((N, 6))

    for i in range(N):
        ind = sample[i]
        u = correspondences[ind][0]
        v = correspondences[ind][1]

        A[i, :] = [
            u[0] * v[0] - u[1] * v[1],
            u[0] * v[1] + u[1] * v[0],
            u[2] * v[0],
            u[2] * v[1],
            u[0] * v[2],
            u[1] * v[2]
        ]
    
    Q, _ = np.linalg.qr(A.T, mode='complete')
    B = Q[:6, 3:6]
    return B

import numpy as np

def compute_matrix(B):
    # B는 (6, 10) 행렬로 가정합니다.
    B = np.array(B)
    
    # 변수 선언
    t2 = B[0, 0] * B[0, 0]
    t3 = 2 * t2
    t4 = B[1, 0] * B[1, 0]
    t5 = 2 * t4
    t6 = B[3, 0] * B[3, 0]
    t7 = 2 * t6
    t8 = t3 + t5 + t7
    t9 = B[2, 0] * B[2, 0]
    t10 = B[4, 0] * B[4, 0]
    t11 = B[5, 0] * B[5, 0]
    t12 = t3 + t5 + t6 + t9 + t10 + t11
    t13 = 4 * B[0, 0] * B[0, 1]
    t14 = 4 * B[1, 0] * B[1, 1]
    t15 = 2 * B[0, 0] * B[5, 0]
    t45 = 2 * B[1, 0] * B[4, 0]
    t16 = t15 - t45
    t17 = 2 * B[2, 0] * B[2, 1]
    t18 = 2 * B[3, 0] * B[3, 1]
    t19 = 2 * B[4, 0] * B[4, 1]
    t20 = 2 * B[5, 0] * B[5, 1]
    t21 = t13 + t14 + t17 + t18 + t19 + t20
    t22 = B[0, 1] * B[0, 1]
    t23 = 2 * t22
    t24 = B[1, 1] * B[1, 1]
    t25 = 2 * t24
    t26 = B[3, 1] * B[3, 1]
    t27 = 2 * B[0, 0] * B[5, 1]
    t28 = 2 * B[0, 1] * B[5, 0]
    t51 = 2 * B[1, 0] * B[4, 1]
    t52 = 2 * B[1, 1] * B[4, 0]
    t29 = t27 + t28 - t51 - t52
    t30 = 4 * B[3, 0] * B[3, 1]
    t31 = t13 + t14 + t30
    t32 = B[0, 0] * B[2, 1]
    t33 = B[0, 1] * B[2, 0]
    t34 = t32 + t33
    t35 = 2 * t26
    t36 = t23 + t25 + t35
    t37 = B[2, 1] * B[2, 1]
    t38 = B[4, 1] * B[4, 1]
    t39 = B[5, 1] * B[5, 1]
    t40 = t23 + t25 + t26 + t37 + t38 + t39
    t41 = 2 * B[0, 1] * B[5, 1]
    t76 = 2 * B[1, 1] * B[4, 1]
    t42 = t41 - t76
    t43 = 4 * B[0, 0] * B[0, 2]
    t44 = 4 * B[1, 0] * B[1, 2]
    t46 = 2 * B[2, 0] * B[2, 2]
    t47 = 2 * B[3, 0] * B[3, 2]
    t48 = 2 * B[4, 0] * B[4, 2]
    t49 = 2 * B[5, 0] * B[5, 2]
    t50 = t43 + t44 + t46 + t47 + t48 + t49
    t53 = 2 * B[0, 0] * B[5, 2]
    t54 = 2 * B[0, 2] * B[5, 0]
    t82 = 2 * B[1, 0] * B[4, 2]
    t83 = 2 * B[1, 2] * B[4, 0]
    t55 = t53 + t54 - t82 - t83
    t56 = 4 * B[3, 0] * B[3, 2]
    t57 = t43 + t44 + t56
    t58 = 4 * B[0, 1] * B[0, 2]
    t59 = 4 * B[1, 1] * B[1, 2]
    t60 = B[0, 0] * B[2, 2]
    t61 = B[0, 2] * B[2, 0]
    t62 = t60 + t61
    t63 = 2 * B[2, 1] * B[2, 2]
    t64 = 2 * B[3, 1] * B[3, 2]
    t65 = 2 * B[4, 1] * B[4, 2]
    t66 = 2 * B[5, 1] * B[5, 2]
    t67 = t58 + t59 + t63 + t64 + t65 + t66
    t68 = 2 * B[0, 1] * B[5, 2]
    t69 = 2 * B[0, 2] * B[5, 1]
    t90 = 2 * B[1, 1] * B[4, 2]
    t91 = 2 * B[1, 2] * B[4, 1]
    t70 = t68 + t69 - t90 - t91
    t71 = 4 * B[3, 1] * B[3, 2]
    t72 = t58 + t59 + t71
    t73 = B[0, 1] * B[2, 2]
    t74 = B[0, 2] * B[2, 1]
    t75 = t73 + t74
    t77 = B[0, 2] * B[0, 2]
    t78 = 2 * t77
    t79 = B[1, 2] * B[1, 2]
    t80 = 2 * t79
    t81 = B[3, 2] * B[3, 2]
    t84 = 2 * t81
    t85 = t78 + t80 + t84
    t86 = B[2, 2] * B[2, 2]
    t87 = B[4, 2] * B[4, 2]
    t88 = B[5, 2] * B[5, 2]
    t89 = t78 + t80 + t81 + t86 + t87 + t88
    t92 = 2 * B[0, 2] * B[5, 2]
    t94 = 2 * B[1, 2] * B[4, 2]
    t93 = t92 - t94
    t95 = 2 * t10
    t96 = 2 * t11
    t97 = t95 + t96
    t98 = 2 * B[0, 0] * B[4, 0]
    t99 = 2 * B[1, 0] * B[5, 0]
    t100 = t98 + t99
    t101 = 2 * B[0, 0] * B[4, 1]
    t102 = 2 * B[0, 1] * B[4, 0]
    t103 = 2 * B[1, 0] * B[5, 1]
    t104 = 2 * B[1, 1] * B[5, 0]
    t105 = t101 + t102 + t103 + t104
    t106 = 4 * B[4, 0] * B[4, 1]
    t107 = 4 * B[5, 0] * B[5, 1]
    t108 = t106 + t107
    t109 = 2 * B[0, 0] * B[4, 2]
    t110 = 2 * B[0, 2] * B[4, 0]
    t111 = 2 * B[1, 0] * B[5, 2]
    t112 = 2 * B[1, 2] * B[5, 0]
    t113 = t109 + t110 + t111 + t112
    t114 = 2 * B[2, 0] * B[4, 0]
    t115 = 2 * B[2, 1] * B[4, 1]
    t116 = 2 * B[2, 2] * B[4, 2]
    t117 = t114 + t115 + t116
    t118 = 2 * B[2, 0] * B[5, 0]
    t119 = 2 * B[2, 1] * B[5, 1]
    t120 = 2 * B[2, 2] * B[5, 2]
    t121 = t118 + t119 + t120
    t122 = 4 * B[0, 1] * B[0, 2]
    t123 = 4 * B[1, 1] * B[1, 2]
    t124 = t122 + t123
    t125 = 4 * B[2, 0] * B[2, 2]
    t126 = 4 * B[3, 0] * B[3, 2]
    t127 = t125 + t126
    t128 = 4 * B[2, 1] * B[2, 2]
    t129 = 4 * B[3, 1] * B[3, 2]
    t130 = t128 + t129
    t131 = 4 * B[4, 0] * B[4, 2]
    t132 = 4 * B[5, 0] * B[5, 2]
    t133 = t131 + t132
    t134 = 4 * B[4, 1] * B[4, 2]
    t135 = 4 * B[5, 1] * B[5, 2]
    t136 = t134 + t135
    t137 = 2 * (B[0, 0] * B[1, 0] + B[0, 1] * B[1, 1] + B[0, 2] * B[1, 2])
    t138 = 2 * (B[2, 0] * B[3, 0] + B[2, 1] * B[3, 1] + B[2, 2] * B[3, 2])
    t139 = 2 * (B[4, 0] * B[5, 0] + B[4, 1] * B[5, 1] + B[4, 2] * B[5, 2])
    
    # 결과 행렬 C 생성
    C = np.zeros((6, 10))

    C[0, 0] = t8
    C[0, 1] = t21
    C[0, 2] = t36
    C[0, 3] = t40
    C[0, 4] = t42
    C[0, 5] = t50
    C[0, 6] = t55
    C[0, 7] = t67
    C[0, 8] = t70
    C[0, 9] = t72
    
    C[1, 0] = t97
    C[1, 1] = t100
    C[1, 2] = t105
    C[1, 3] = t108
    C[1, 4] = t113
    C[1, 5] = t117
    C[1, 6] = t121
    C[1, 7] = t124
    C[1, 8] = t127
    C[1, 9] = t130

    C[2, 0] = t133
    C[2, 1] = t136
    C[2, 2] = t139
    
    return C

# 예시 행렬 B
B = [
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36]
]

# 함수 호출
C = compute_matrix(B)
print(C)


import numpy as np
from scipy.linalg import eig, solve, norm
from scipy.optimize import minimize

def minimal_solver(C):
    from numpy.linalg import matrix_rank

    # Compute matrix G
    C_sub_0 = C[:, :6]
    C_sub_1 = C[:, 6:10]
    G = solve(C_sub_0, C_sub_1)
    
    # Initialize matrix M
    M = np.zeros((4, 4))
    M[0, :] = -G[2, :]
    M[1, :] = -G[4, :]
    M[2, :] = -G[5, :]
    M[3, 1] = 1

    # Eigen decomposition
    eigvals, eigvecs = eig(M)
    
    results = []
    for i in range(4):
        if abs(np.imag(eigvals[i])) > 1e-12:
            continue
        
        bsoln = np.real(eigvecs[1:4, i])
        psoln = np.dot(C[:, :6], bsoln)
        
        Esoln = np.array([
            [psoln[0], psoln[1], psoln[2]],
            [psoln[1], -psoln[0], psoln[3]],
            [psoln[4], psoln[5], 0]
        ])
        
        Esoln /= norm(Esoln)
        results.append(Esoln)
    
    return results

def non_minimal_solver(sample, C):
    Es = minimal_solver(C)
    if not Es:
        return 0, None
    
    def evaluate_model_on_point(E, idx):
        # Placeholder function for evaluating the model
        # Replace with actual implementation
        return np.random.rand()
    
    best_score = float('inf')
    best_ind = -1
    for i, E in enumerate(Es):
        score = sum(evaluate_model_on_point(E, idx) for idx in sample)
        if score < best_score:
            best_score = score
            best_ind = i
    
    return 1, Es[best_ind]

def least_squares(sample, correspondences, E):
    def decompose_spherical_essential_matrix(E, inward):
        # Placeholder function for matrix decomposition
        # Replace with actual implementation
        r = np.zeros(3)
        t = np.zeros(3)
        return r, t
    
    def make_spherical_essential_matrix(r, inward, E):
        # Placeholder function for matrix creation
        # Replace with actual implementation
        pass
    
    def sampson_error(params, u, v):
        # Placeholder function for Sampson error
        # Replace with actual implementation
        return np.sum(params**2)

    # Initial parameters
    r0 = np.array([0, 0, 0])
    t0 = np.array([0, 0, -1])
    if inward:
        t0[2] = 1
    
    r, t = decompose_spherical_essential_matrix(E, inward)
    r1 = np.copy(r)
    t1 = np.array([0, 0, -1])
    if inward:
        t1[2] = 1
    
    u = [correspondences[idx][0] for idx in sample]
    v = [correspondences[idx][1] for idx in sample]
    
    def objective(params):
        r0, t0, r1, t1 = params[:3], params[3:6], params[6:9], params[9:]
        total_error = 0
        for i in range(len(sample)):
            total_error += sampson_error(np.concatenate([r0, t0, r1, t1]), u[i], v[i])
        return total_error
    
    initial_params = np.concatenate([r0, t0, r1, t1])
    result = minimize(objective, initial_params, method='trust-constr')
    optimal_params = result.x
    
    r0_opt, t0_opt, r1_opt, t1_opt = optimal_params[:3], optimal_params[3:6], optimal_params[6:9], optimal_params[9:]
    r0_opt = np.copy(r0_opt)
    t0_opt = np.copy(t0_opt)
    r1_opt = np.copy(r1_opt)
    t1_opt = np.copy(t1_opt)
    
    make_spherical_essential_matrix(r1_opt, inward, E)

# Example usage
C = np.random.rand(6, 10)
sample = [0, 1, 2]  # Example sample indices
correspondences = [(np.random.rand(3), np.random.rand(3)) for _ in range(10)]
E = np.random.rand(3, 3)  # Placeholder for the essential matrix
inward = False

# Call functions
print(minimal_solver(C))
print(non_minimal_solver(sample, C))
least_squares(sample, correspondences, E)

