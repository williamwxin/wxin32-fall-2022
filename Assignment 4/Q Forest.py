import time

import matplotlib.pyplot as plt
import numpy as np
import hiive.mdptoolbox as mdp
import seaborn as sns



P,R = mdp.example.forest(16, p=0.2)
ql = QLearning(P,R,0.95, alpha=0.25, alpha_decay=.99995, epsilon=1,epsilon_decay=0.99995,n_iter=1000000, epsilon_min=0.01)
ql.verbose=True
qvs_fo_small_timed, qps_fo_small_timed, dqvs_fo_small_timed, dqps_fo_small_timed, dqs_fo_small_timed, iters_fo_small_timed, times_fo_small_timed = ql.run(y_step=100)
scatter_fo_small_timed, labels_fo_small_timed = vs_scatter_thresh(qvs_fo_small_timed, y_step=500, sort=True)

P,R = mdp.example.forest(16, p=0.2)
ql = QLearning(P,R,0.95, alpha=0.25, alpha_decay=.99995, epsilon=1,epsilon_decay=0.99995,n_iter=100000, epsilon_min=0.01)
ql.verbose=True
qvs_fo_small, qps_fo_small, dqvs_fo_small, dqps_fo_small, dqs_fo_small, iters_fo_small, times_fo_small = ql.run(y_step=1)
scatter_fo_small, labels_fo_small = vs_scatter_thresh(qvs_fo_small, y_step=500, sort=True)

P,R = mdp.example.forest(2500, p=0.0001)
ql = QLearning(P,R,0.9995, alpha=1, alpha_decay=.999995, epsilon=1,epsilon_decay=0.999995,n_iter=1000000, epsilon_min=0.01)
ql.verbose=True
qvs_fo_big, qps_fo_big, dqvs_fo_big, dqps_fo_big, dqs_fo_big, iters_fo_big, times_fo_big = ql.run(y_step=1000)
scatter_fo_big, labels_fo_big = qs_scatter_thresh(qvs_fo_big,iters_fo_big, x_step=100, thresh=.95, sort=False)

P,R = mdp.example.forest(2500, p=0.0001)
ql = QLearning(P,R,0.9995, alpha=1, alpha_decay=.999995, epsilon=1,epsilon_decay=0.999995,n_iter=1000000, epsilon_min=0.01)
ql.Q[:] = 5000
ql.verbose=True
qvs_fo_big_happy, qps_fo_big_happy, dqvs_fo_big_happy, dqps_fo_big_happy, dqs_fo_big_happy, iters_fo_big_happy, times_fo_big_happy = ql.run(y_step=1000)
scatter_fo_big_happy, labels_fo_big_happy = vs_scatter_thresh(qvs_fo_big_happy, x_step=100, y_step=5, thresh=.95, sort=True)


vi = ValueIteration(P,R,0.95, epsilon=.001)
vi.verbose=True
vi.max_iter = 10000
vvs_fo_small, vps_fo_small, vvs_d_fo_small, vps_d_fo_small = vi.run_vp()

alpha_decays = [0.9,0.95,0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999]
eps_decays = [0.9,0.95,0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999]
vi = ValueIteration(P,R,0.95, epsilon=.0001)
vi.verbose=True
vi.max_iter = 10000
vvs_fo_small, vps_fo_small, vvs_d_fo_small, vps_d_fo_small = vi.run_vp()

ae_fo_small_100k = np.zeros((len(alpha_decays),len(eps_decays),3))
avg=2
for x,a in enumerate(alpha_decays):
    for y,e in enumerate(eps_decays):
        for c in range(avg):
            start = time.time()
            ql = QLearning(P, R, 0.95, alpha=1, alpha_decay=a, epsilon=1, epsilon_decay=a, n_iter=100000)
            vs, ps, dvs, dps, dqs, alphas, eps = ql.run()
            end = time.time()-start
            vs_diff = np.mean(np.abs(vs[-1,:]-vvs_fo_small[-1,:]))
            ps_diff = np.mean(ps[-1,:] != vps_fo_small[-1,:])
            ae_fo_small_100k[x,y,0] += end/avg
            ae_fo_small_100k[x, y, 1] += vs_diff / avg
            ae_fo_small_100k[x, y, 2] += ps_diff / avg
        print(a,e)

np.savetxt('AE0 Forest 100k.csv',ae_fo_small_100k[:,:,0],delimiter=',')
np.savetxt('AE1 Forest 100k.csv',ae_fo_small_100k[:,:,1],delimiter=',')
np.savetxt('AE2 Forest 100k.csv',ae_fo_small_100k[:,:,2],delimiter=',')


# ae_fo_small_100k_scatter_axes = [[],[]]
# for x,a in enumerate(alpha_decays):
#     for y,e in enumerate(eps_decays):
#         ae_fo_small_100k_scatter_axes[0].append(a)
#         ae_fo_small_100k_scatter_axes[1].append(e)
# ae_fo_small_100k_scatter_axes = np.array(ae_fo_small_100k_scatter_axes)
# ae_fo_small_100k_scatter_axes[0] = np.array(ae_fo_small_100k_scatter_axes[0], dtype='str')
# ae_fo_small_100k_scatter_axes[1] = np.array(ae_fo_small_100k_scatter_axes[1], dtype='str')

# ae_fo_small_100k_scatter_vals = [[],[],[]]
# for x,a in enumerate(alpha_decays):
#     for y,e in enumerate(eps_decays):
#         ae_fo_small_100k_scatter_vals[0].append(ae_fo_small_100k[x,y,0])
#         ae_fo_small_100k_scatter_vals[1].append(ae_fo_small_100k[x,y,1])
#         ae_fo_small_100k_scatter_vals[2].append(ae_fo_small_100k[x, y, 2])
# ae_fo_small_100k_scatter_vals = np.array(ae_fo_small_100k_scatter_vals)


ae_fo_small_500k = np.zeros((len(alpha_decays),len(eps_decays),3))
avg=2
for x,a in enumerate(alpha_decays):
    for y,e in enumerate(eps_decays):
        for c in range(avg):
            start = time.time()
            ql = QLearning(P, R, 0.95, alpha=1, alpha_decay=a, epsilon=1, epsilon_decay=a, n_iter=500000)
            vs, ps, dvs, dps, dqs, alphas, eps = ql.run()
            end = time.time()-start
            vs_diff = np.mean(np.abs(vs[-1,:]-vvs_fo_small[-1,:]))
            ps_diff = np.mean(ps[-1,:] != vps_fo_small[-1,:])
            ae_fo_small_500k[x,y,0] += end/avg
            ae_fo_small_500k[x, y, 1] += vs_diff / avg
            ae_fo_small_500k[x, y, 2] += ps_diff / avg
        print(a,e)


np.savetxt('AE0 Forest 500k.csv',ae_fo_small_500k[:,:,0],delimiter=',')
np.savetxt('AE1 Forest 500k.csv',ae_fo_small_500k[:,:,1],delimiter=',')
np.savetxt('AE2 Forest 500k.csv',ae_fo_small_500k[:,:,2],delimiter=',')




temp = ae_fo_small_100k[:,:,1]

ps_fo_small, deltas_fo_small = pi.run_non_holes(p_non_holes)
scatter_fo_small, labels_fo_small = ps_scatter_thresh(ps_fo_small, sort=True)

def find_non_holes(fr, g=0.999):
    conversion = OpenAI_MDPToolbox("ForestLake-v1", desc=fr)
    P,R = conversion.P, conversion.R
    vi = ValueIteration(P,R,g, epsilon=0.001)
    vs_fo_big, deltas_fo_big = vi.run()
    return np.where(vs_fo_big[-1,:]!=0)[0]

def fix_p(P):
    P[P!=0] += np.random.rand(np.sum(P!=0))/50000
    temp = np.sum(P, axis=2)
    temp2 = np.tile(temp, len(temp[0,:]))
    temp2 = temp2.reshape((len(temp),len(temp[0]),len(temp[0])))
    temp2 = np.transpose(temp2, (0,2,1))
    P /= temp2
    return P

conversion = OpenAI_MDPToolbox("ForestLake-v1", desc=fr)
p_non_holes = find_non_holes(fr, 0.995)
P,R = conversion.P, conversion.R
P = fix_p(P)

pi = PolicyIteration(P,R,0.95)
ps_fo_big, deltas_fo_big = pi.run_non_holes(p_non_holes)
scatter_fo_big, labels_fo_big = ps_scatter_thresh(ps_fo_big, x_step=1, y_step=1, thresh=.9, sort=True)

axes_temp_x = np.array(list(range(len(alpha_decays)))*len(eps_decays)).reshape((len(alpha_decays),len(eps_decays))).T.flatten()
axes_temp_y = np.array(list(range(len(alpha_decays)))*len(eps_decays))

tempx, tempy = np.meshgrid(range(len(alpha_decays)),range(len(eps_decays)))

temp = ae_fo_small_100k[:,:,1]

widths = [4, 8, 10, 15, 20, 25, 30, 33, 35, 38, 40, 43, 45, 48, 50, 53, 55]
qtps_fo_res = np.zeros((len(widths), 2))
qtps_fo_res[:, 0] = np.square(widths)
avg = 2
for i, w in enumerate(widths):

    for a in range(avg):
        P, R = mdp.example.forest(w*w, p=0.0001)
        ql = QLearning(P, R, 0.9995, alpha=1, alpha_decay=.99995, epsilon=1, epsilon_decay=0.99995, n_iter=10000, epsilon_min=0.01)
        ql.verbose = True
        start = time.time()
        temp = ql.run(y_step=1)
        qtps_fo_res[i, 1] += (time.time() - start) / avg

    print(w)

sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-66)
plt.title('Forest Lake 4x4 PI Values')
plt.title('Forest Lake 4x4 PI Values')
#ax.scatter(range(len(ae_fo_small_100k_scatter_axes[0])), range(len(ae_fo_small_100k_scatter_axes[1])), ae_fo_small_100k_scatter_vals[1], c=ae_fo_small_100k_scatter_vals[1],cmap='jet')
ax.plot_wireframe(tempx.T, tempy.T, ae_fo_small_100k[:,:,2], color='black', linewidth=0.5)
ax.scatter(axes_temp_x, axes_temp_y, ae_fo_small_100k_scatter_vals[2], c=ae_fo_small_100k_scatter_vals[2],cmap='jet')

ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Value')
for x,y,z in labels_fo_small:
    ax.text(x,y,z,z, ha='center', fontsize=10)
ax.tick_params(axis='both', labelsize=9)
ax.text(0,0,max(scatter_fo_small[:,2])*1.5, 'Gamma=0.95', fontsize=11, color='purple')
plt.savefig('PI Forest small values')



sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-66)
plt.title('Forest 16 QL Values')
ax.scatter(scatter_fo_small[:,0], scatter_fo_small[:,1], scatter_fo_small[:,2], c=np.log((scatter_fo_small[:,2]/1000+1)), cmap='jet')
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Max Q')
for x,y,z in labels_fo_small:
    ax.text(x,y,z+1,z, ha='center', fontsize=10)
ax.tick_params(axis='both', labelsize=9)
ax.text(0,0,max(scatter_fo_small[:,2])*1.5, 'Gamma=0.95', fontsize=11, color='purple')
plt.savefig('QL Forest small values')

l = len(dqvs_fo_small)
step = 1000
sns.set()
fig,ax = plt.subplots(figsize=(3.1, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.26)
plt.title('Forest 16 QL Errors (Rolling Avg)')
sns.lineplot(y=moving_average(dqvs_fo_small,n=10000)[range(0,l,step)], x=range(0,len(dqvs_fo_small),step))
plt.xlabel('Iteration')
plt.ylabel('Change in Q (Error)')
ax.tick_params(axis='both', labelsize=9)
plt.savefig('QL Forest small error')


sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(18,-65)
plt.title('Forest 2500 QL Values')
ax.scatter(scatter_fo_big[:,0], scatter_fo_big[:,1], scatter_fo_big[:,2], c=np.log((scatter_fo_big[:,2]/1000+1)), cmap='jet')
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Max Q')
for x,y,z in labels_fo_big:
    ax.text(x,y,z+100,np.round(z,1), ha='center', fontsize=10)
ax.tick_params(axis='both', labelsize=9)
ax.set_zlim([0,1500])
ax.text(0,0,max(scatter_fo_big[:,2])*2, 'Gamma=0.9995', fontsize=11, color='purple')
plt.ticklabel_format(axis='both', style='plain')
plt.savefig('QL Forest big values')

l = len(dqvs_fo_big)
step = 1
sns.set()
fig,ax = plt.subplots(figsize=(3.3, 2.7))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.265)
plt.title('Forest 2500 QL Errors (Rolling Avg)')
sns.lineplot(y=moving_average(dqvs_fo_big,n=50000)[range(0,l,step)], x=range(0,len(dqvs_fo_big),step))
plt.xlabel('Iteration')
plt.ylabel('Change in Q (Error)')
ax.tick_params(axis='both', labelsize=9)
plt.ticklabel_format(axis='both', style='plain')
plt.savefig('QL Forest big error')

sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(18,-65)
plt.title('Forest 4x4 QL Values')
ax.scatter(scatter_fo_big_happy[:,0], scatter_fo_big_happy[:,1]*1000, scatter_fo_big_happy[:,2], c=np.log((scatter_fo_big_happy[:,2]/1000+1)), cmap='jet')
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Max Q')
for x,y,z in labels_fo_big_happy:
    ax.text(x,y*1000,z+1,z, ha='center', fontsize=10)
ax.tick_params(axis='both', labelsize=9)
ax.text(0,0,max(scatter_fo_big_happy[:,2])*1.5, 'Gamma=0.95', fontsize=11, color='purple')
plt.savefig('QL Forest big_happy values')

