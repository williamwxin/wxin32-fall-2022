import time

import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import hiive.mdptoolbox as mdp

def frozen_generator(p=0.125, width=16, fixer=False):
    np.random.seed(1)
    arr = ['S']
    for i in range(width*width-2):
        if np.random.random() >= p:
            arr.append('F')
        else:
            arr.append('H')
    arr.append('G')
    arr = np.array(arr).reshape((width,width))
    if fixer:
        arr[:,0] = 'F'
        arr[1,1] = 'F'
        arr[0,:] = 'F'
        arr[-1,:] = 'F'
        arr[-2,-2] = 'F'
        arr[:,-1] = 'F'
        arr[0,0] = 'S'
        arr[-1,-1] = 'G'
    return arr

def ps_scatter_thresh(vs, x_step=1, y_step=1, thresh = 1, sort=True):
    v_s = np.array(vs)
    end = len(v_s) - 1
    first_label = v_s[-1,0]
    if sort:
        temp = v_s[-1, :].copy()
        order = np.argsort(temp)
        v_s = v_s[:,order]

    scatter = []
    for x in range(0,int(len(v_s[0,:])*thresh),x_step):
        for y in range(0,len(v_s[:,0]),y_step):
            scatter.append([x, y, v_s[y,x]])
    for x in range(int(len(v_s[0,:])*thresh),len(v_s[0,:]),1):
        for y in range(0,len(v_s[:,0]),y_step):
            scatter.append([x, y, v_s[y,x]])
    scatter = np.array(scatter)
    median_s = int(len(v_s[0,:])/2)-1

    labels = [(0,end,first_label), (median_s,end,v_s[-1,median_s]), (len(v_s[0,:])-1,end,v_s[-1,-1])]
    labels = np.array(labels)
    return scatter, np.round(labels,3)

def non_holes(fr):
    fr_flat = fr.flatten()
    fr_non_holes = np.array(range(len(fr_flat)))[fr_flat!='H'][0:-1]
    return fr_non_holes

def vps_scatter_thresh(vs, ps, x_step=1, y_step=1, thresh = 1, sort=True):
    v_s = np.array(vs.copy())
    p_s = np.array(ps.copy())
    end = len(v_s) - 1
    first_label = v_s[-1,0]
    if sort:
        temp = v_s[-1, :].copy()
        order = np.argsort(temp)
        v_s = v_s[:,order]
        p_s = p_s[:,order]

    scatter = []
    scatter_p = []
    for x in range(0,int(len(v_s[0,:])*thresh),x_step):
        for y in range(0,len(v_s[:,0]),y_step):
            scatter.append([x, y, v_s[y,x]])
            scatter_p.append([x, y, p_s[y, x]])
    for x in range(int(len(v_s[0,:])*thresh),len(v_s[0,:]),1):
        for y in range(0,len(v_s[:,0]),y_step):
            scatter.append([x, y, v_s[y,x]])
            scatter_p.append([x, y, p_s[y, x]])
    scatter = np.array(scatter)
    scatter_p = np.array(scatter_p)
    median_s = int(len(v_s[0,:])/2)-1

    labels = [(0,end,first_label), (median_s,end,v_s[-1,median_s]), (len(v_s[0,:])-1,end,v_s[-1,-1])]
    labels = np.array(labels)
    return scatter, scatter_p, np.round(labels,3)
def find_non_holes(fr, g=0.999):
    conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr)
    P,R = conversion.P, conversion.R
    vi = ValueIteration(P,R,g, epsilon=0.001)
    vs_fl_big, deltas_fl_big = vi.run()
    return np.where(vs_fl_big[-1,:]!=0)[0]
def fix_p(P):
    P[P!=0] += np.random.rand(np.sum(P!=0))/50000
    temp = np.sum(P, axis=2)
    temp2 = np.tile(temp, len(temp[0,:]))
    temp2 = temp2.reshape((len(temp),len(temp[0]),len(temp[0])))
    temp2 = np.transpose(temp2, (0,2,1))
    P /= temp2
    return P
def vs_scatter_thresh(vs, x_step=1, y_step=1, thresh = 1, sort=True, opposite=False):
    v_s = np.array(vs)
    end = len(v_s) - 1
    first_label = v_s[-1,0]
    if sort:
        temp = v_s[-1, :].copy()
        order = np.argsort(temp)
        if opposite:
            order = order[::-1]
        v_s = v_s[:,order]

    scatter = []
    for x in range(0,int(len(v_s[0,:])*thresh),x_step):
        for y in range(0,len(v_s[:,0]),y_step):
            scatter.append([x, y, v_s[y,x]])
    for x in range(int(len(v_s[0,:])*thresh),len(v_s[0,:]),1):
        for y in range(0,len(v_s[:,0]),y_step):
            scatter.append([x, y, v_s[y,x]])
    scatter = np.array(scatter)
    median_s = int(len(v_s[0,:])/2)-1

    labels = [(0,end,first_label), (median_s,end,v_s[-1,median_s]), (len(v_s[0,:])-1,end,v_s[-1,-1])]
    labels = np.array(labels)
    return scatter, np.round(labels,3)


vs = qvs_fl_small_timed
iters = iters_fl_small_timed
def qs_scatter_thresh(vs, iters, x_step=1, y_step=1, thresh = 1, sort=True, opposite=False):
    v_s = np.array(vs)
    end = len(v_s) - 1
    first_label = v_s[-1,0]
    if sort:
        temp = v_s[-1, :].copy()
        order = np.argsort(temp)
        if opposite:
            order = order[::-1]
        v_s = v_s[:,order]

    scatter = []
    for x in range(0,int(len(v_s[0,:])*thresh),x_step):
        for i,y in enumerate(iters):
            scatter.append([x, y, v_s[i,x]])
    for x in range(int(len(v_s[0,:])*thresh),len(v_s[0,:]),1):
        for i,y in enumerate(iters):
            scatter.append([x, y, v_s[i,x]])
    scatter = np.array(scatter)
    median_s = int(len(v_s[0,:])/2)-1

    labels = [(0,iters[-1],first_label), (median_s,iters[-1],v_s[-1,median_s]), (len(v_s[0,:])-1,iters[-1],v_s[-1,-1])]
    labels = np.array(labels)
    return scatter, np.round(labels,3)




def moving_average(a, n=1000) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n

fr = frozen_generator(width=50, fixer=False)
fr4 = frozen_generator(width=4)


conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr4)
p_non_holes = find_non_holes(fr4, 0.995)
P,R = conversion.P, conversion.R
P = fix_p(P)
ql = QLearning(P,R,0.95, alpha=0.25, alpha_decay=.99995, epsilon=1,epsilon_decay=0.99995,n_iter=100000, epsilon_min=0.01)
ql.verbose=True
qvs_fl_small, qps_fl_small, dqvs_fl_small, dqps_fl_small, dqs_fl_small, iters_fl_small, times_fl_small = ql.run(y_step=1)
scatter_fl_small, labels_fl_small = vs_scatter_thresh(qvs_fl_small[:,find_non_holes(fr4)], y_step=500, sort=True)

conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr4)
p_non_holes = find_non_holes(fr4, 0.995)
P,R = conversion.P, conversion.R
P = fix_p(P)
ql = QLearning(P,R,0.95, alpha=0.25, alpha_decay=.99995, epsilon=1,epsilon_decay=0.99995,n_iter=1000000, epsilon_min=0.01)
ql.verbose=True
qvs_fl_small_timed, qps_fl_small_timed, dqvs_fl_small_timed, dqps_fl_small_timed, dqs_fl_small_timed, iters_fl_small_timed, times_fl_small_timed = ql.run(y_step=1000)
scatter_fl_small_timed, labels_fl_small_timed = qs_scatter_thresh(qvs_fl_small_timed[:,find_non_holes(fr4)], iters_fl_small_timed, y_step=500, sort=True)
# np.mean(np.abs(vs[-1,:]-vvs_fl_small[-1,:]))
# np.mean(ps[-1,:] != vps_fl_small[-1,:])

conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr)
p_non_holes = find_non_holes(fr, 0.995)
P,R = conversion.P, conversion.R
P = fix_p(P)
ql = QLearning(P,R,0.9995, alpha=1, alpha_decay=.999995, epsilon=1,epsilon_decay=0.999995,n_iter=1000000, epsilon_min=0.01)
ql.verbose=True
qvs_fl_big, qps_fl_big, dqvs_fl_big, dqps_fl_big, dqs_fl_big, iters_fl_big, times_fl_big = ql.run(y_step=1000)
scatter_fl_big, labels_fl_big = qs_scatter_thresh(qvs_fl_big, iters_fl_big, x_step=100, thresh=.95, sort=True)

conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr)
p_non_holes = find_non_holes(fr, 0.995)
P,R = conversion.P, conversion.R
P = fix_p(P)
ql = QLearning(P,R,0.9995, alpha=1, alpha_decay=.999995, epsilon=1,epsilon_decay=0.999995,n_iter=1000000, epsilon_min=0.01)
ql.Q[p_non_holes,:] = 1
qvs_fl_big_happy, qps_fl_big_happy, dqvs_fl_big_happy, dqps_fl_big_happy, dqs_fl_big_happy, iters_fl_big_happy, times_fl_big_happy = ql.run(y_step=1000)
scatter_fl_big_happy, labels_fl_big_happy = vs_scatter_thresh(qvs_fl_big_happy, x_step=100, y_step=5, thresh=.95, sort=True)


vi = ValueIteration(P,R,0.95, epsilon=.001)
vi.verbose=True
vi.max_iter = 10000
vvs_fl_small, vps_fl_small, vvs_d_fl_small, vps_d_fl_small = vi.run_vp()

alpha_decays = [0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999, 0.999995, 0.999999]
eps_decays = [0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999, 0.999995, 0.999999]
vi = ValueIteration(P,R,0.95, epsilon=.0001)
vi.verbose=True
vi.max_iter = 10000
vvs_fl_small, vps_fl_small, vvs_d_fl_small, vps_d_fl_small = vi.run_vp()

ae_fl_small_100k = np.zeros((len(alpha_decays),len(eps_decays),3))
avg=3
for x,a in enumerate(alpha_decays):
    for y,e in enumerate(eps_decays):
        for c in range(avg):
            start = time.time()
            ql = QLearning(P, R, 0.95, alpha=0.25, alpha_decay=a, epsilon=1, epsilon_decay=a, n_iter=100000)
            vs, ps, dvs, dps, dqs, alphas, eps = ql.run(verbose=False, y_step=1)
            end = time.time()-start
            vs_diff = np.mean(np.abs(vs[-1,:]-vvs_fl_small[-1,:]))
            ps_diff = np.mean(ps[-1,:] != vps_fl_small[-1,:])
            ae_fl_small_100k[x,y,0] += end/avg
            ae_fl_small_100k[x, y, 1] += vs_diff / avg
            ae_fl_small_100k[x, y, 2] += ps_diff / avg
        print(a,e)

np.savetxt('AE0 Frozen 100k.csv',ae_fl_small_100k[:,:,0],delimiter=',')
np.savetxt('AE1 Frozen 100k.csv',ae_fl_small_100k[:,:,1],delimiter=',')
np.savetxt('AE2 Frozen 100k.csv',ae_fl_small_100k[:,:,2],delimiter=',')

if True:
    P, R = mdp.example.forest(16, p=0.2)
    vi = ValueIteration(P, R, 0.95, epsilon=.0001)
    vi.verbose = True
    vi.max_iter = 10000
    vvs_fo_small, vps_fo_small, vvs_d_fo_small, vps_d_fo_small = vi.run_vp()

    ae_fo_small_100k = np.zeros((len(alpha_decays), len(eps_decays), 3))
    avg = 3
    for x, a in enumerate(alpha_decays):
        for y, e in enumerate(eps_decays):
            for c in range(avg):
                start = time.time()
                ql = QLearning(P, R, 0.95, alpha=0.25, alpha_decay=a, epsilon=1, epsilon_decay=a, n_iter=100000)
                vs, ps, dvs, dps, dqs, alphas, eps = ql.run(verbose=False, y_step=1)
                end = time.time() - start
                vs_diff = np.mean(np.abs(vs[-1, :] - vvs_fo_small[-1, :]))
                ps_diff = np.mean(ps[-1, :] != vps_fo_small[-1, :])
                ae_fo_small_100k[x, y, 0] += end / avg
                ae_fo_small_100k[x, y, 1] += vs_diff / avg
                ae_fo_small_100k[x, y, 2] += ps_diff / avg
            print(a, e)

    np.savetxt('AE0 Forest 100k.csv', ae_fo_small_100k[:, :, 0], delimiter=',')
    np.savetxt('AE1 Forest 100k.csv', ae_fo_small_100k[:, :, 1], delimiter=',')
    np.savetxt('AE2 Forest 100k.csv', ae_fo_small_100k[:, :, 2], delimiter=',')

# ae_fl_small_100k_scatter_axes = [[],[]]
# for x,a in enumerate(alpha_decays):
#     for y,e in enumerate(eps_decays):
#         ae_fl_small_100k_scatter_axes[0].append(a)
#         ae_fl_small_100k_scatter_axes[1].append(e)
# ae_fl_small_100k_scatter_axes = np.array(ae_fl_small_100k_scatter_axes)
# ae_fl_small_100k_scatter_axes[0] = np.array(ae_fl_small_100k_scatter_axes[0], dtype='str')
# ae_fl_small_100k_scatter_axes[1] = np.array(ae_fl_small_100k_scatter_axes[1], dtype='str')

# ae_fl_small_100k_scatter_vals = [[],[],[]]
# for x,a in enumerate(alpha_decays):
#     for y,e in enumerate(eps_decays):
#         ae_fl_small_100k_scatter_vals[0].append(ae_fl_small_100k[x,y,0])
#         ae_fl_small_100k_scatter_vals[1].append(ae_fl_small_100k[x,y,1])
#         ae_fl_small_100k_scatter_vals[2].append(ae_fl_small_100k[x, y, 2])
# ae_fl_small_100k_scatter_vals = np.array(ae_fl_small_100k_scatter_vals)


def find_non_holes(fr, g=0.999):
    conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr)
    P,R = conversion.P, conversion.R
    vi = ValueIteration(P,R,g, epsilon=0.001)
    vs_fl_big, deltas_fl_big = vi.run()
    return np.where(vs_fl_big[-1,:]!=0)[0]

def fix_p(P):
    P[P!=0] += np.random.rand(np.sum(P!=0))/50000
    temp = np.sum(P, axis=2)
    temp2 = np.tile(temp, len(temp[0,:]))
    temp2 = temp2.reshape((len(temp),len(temp[0]),len(temp[0])))
    temp2 = np.transpose(temp2, (0,2,1))
    P /= temp2
    return P

conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr)
p_non_holes = find_non_holes(fr, 0.995)
P,R = conversion.P, conversion.R
P = fix_p(P)

pi = PolicyIteration(P,R,0.95)
ps_fl_big, deltas_fl_big = pi.run_non_holes(p_non_holes)
scatter_fl_big, labels_fl_big = ps_scatter_thresh(ps_fl_big, x_step=1, y_step=1, thresh=.9, sort=True)

axes_temp_x = np.array(list(range(len(alpha_decays)))*len(eps_decays)).reshape((len(alpha_decays),len(eps_decays))).T.flatten()
axes_temp_y = np.array(list(range(len(alpha_decays)))*len(eps_decays))

tempx, tempy = np.meshgrid(range(len(alpha_decays)),range(len(eps_decays)))

temp = ae_fl_small_100k[:,:,1]




widths = [4, 8, 10, 15, 20, 25, 30, 33, 35, 38, 40, 43, 45, 48, 50, 53, 55]
qtps_fl_res = np.zeros((len(widths),2))
qtps_fl_res[:,0] = np.square(widths)
avg= 2
for i, w in enumerate(widths):

    for a in range(avg):
        fr_i = frozen_generator(width=w, fixer=True)
        conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr_i)
        p_non_holes = find_non_holes(fr_i, 0.995)
        P, R = conversion.P, conversion.R
        P = fix_p(P)

        ql = QLearning(P, R, 0.995, alpha=0.25, alpha_decay=.99995, epsilon=1, epsilon_decay=0.99995, n_iter=10000, epsilon_min=0.01)
        ql.verbose = True
        start = time.time()
        temp = ql.run(y_step=1)
        qtps_fl_res[i,1] += (time.time() - start)/avg

    print(w)


sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-66)
plt.title('Frozen Lake 4x4 PI Values')
plt.title('Frozen Lake 4x4 PI Values')
#ax.scatter(range(len(ae_fl_small_100k_scatter_axes[0])), range(len(ae_fl_small_100k_scatter_axes[1])), ae_fl_small_100k_scatter_vals[1], c=ae_fl_small_100k_scatter_vals[1],cmap='jet')
ax.plot_wireframe(tempx.T, tempy.T, ae_fl_small_100k[:,:,2], color='black', linewidth=0.5)
ax.scatter(axes_temp_x, axes_temp_y, ae_fl_small_100k_scatter_vals[2], c=ae_fl_small_100k_scatter_vals[2],cmap='jet')

ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Value')
for x,y,z in labels_fl_small:
    ax.text(x,y,z,z, ha='center', fontsize=10)
ax.tick_params(axis='both', labelsize=9)
ax.text(0,0,max(scatter_fl_small[:,2])*1.5, 'Gamma=0.95', fontsize=11, color='purple')
plt.savefig('PI Frozen small values')




sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-66)
plt.title('Frozen Lake 4x4 QL Values')
ax.scatter(scatter_fl_small[:,0], scatter_fl_small[:,1], scatter_fl_small[:,2], c=np.log((scatter_fl_small[:,2]*10+1)), cmap='jet')
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Max Q')
for x,y,z in labels_fl_small:
    ax.text(x,y,z+.1,z, ha='center', fontsize=10)
ax.tick_params(axis='both', labelsize=9)
ax.text(0,0,max(scatter_fl_small[:,2])*1.5, 'Gamma=0.95', fontsize=11, color='purple')
plt.savefig('QL Frozen small values')

l = len(dqvs_fl_small)
step = 1000
sns.set()
fig,ax = plt.subplots(figsize=(3.1, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.26)
plt.title('Frozen 4x4 QL Errors (Rolling Avg)')
sns.lineplot(y=moving_average(dqvs_fl_small,n=1000)[range(0,l,step)], x=range(0,len(dqvs_fl_small),step))
plt.xlabel('Iteration')
plt.ylabel('Change in Q (Error)')
ax.tick_params(axis='both', labelsize=9)
plt.ticklabel_format(axis='y', style='plain')
plt.savefig('QL Frozen small error')


sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(18,-65)
plt.title('Frozen 50x50 QL Values')
ax.scatter(scatter_fl_big[:,0], scatter_fl_big[:,1], scatter_fl_big[:,2], c=np.log((scatter_fl_big[:,2]/1000+1)), cmap='jet')
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Max Q')
for x,y,z in labels_fl_big:
    ax.text(x,y,z+.1,z, ha='center', fontsize=10)
ax.tick_params(axis='both', labelsize=9)
ax.tick_params(axis='y', labelsize=8)
plt.ticklabel_format(axis='y', style='plain')
ax.text(0,0,max(scatter_fl_big[:,2])*1.5, 'Gamma=0.995', fontsize=11, color='purple')
plt.savefig('QL Frozen big values')

l = len(dqvs_fl_big)
step = 1
sns.set()
fig,ax = plt.subplots(figsize=(3.3, 2.7))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.265)
plt.title('Frozen 50x50 QL Errors (Rolling Avg)')
sns.lineplot(y=moving_average(dqvs_fl_big,n=50000)[range(0,l,step)], x=range(0,len(dqvs_fl_big),step))
plt.xlabel('Iteration')
plt.ylabel('Change in Q (Error)')
ax.tick_params(axis='both', labelsize=9)
plt.ticklabel_format(axis='both', style='plain')
plt.savefig('QL Frozen big error')


sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(18,-65)
plt.title('Forest 4x4 QL Values')
ax.scatter(scatter_fl_big_happy[:,0], scatter_fl_big_happy[:,1]*1000, scatter_fl_big_happy[:,2], c=np.log((scatter_fl_big_happy[:,2]/1000+1)), cmap='jet')
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Max Q')
for x,y,z in labels_fl_big_happy:
    ax.text(x,y*1000,z+1,z, ha='center', fontsize=10)
ax.tick_params(axis='both', labelsize=9)
ax.text(0,0,max(scatter_fl_big_happy[:,2])*1.5, 'Gamma=0.95', fontsize=11, color='purple')
plt.savefig('QL Forest big_happy values')




sns.set()
fig,ax = plt.subplots(figsize=(2.5, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.26)
plt.title('Forest 4x4 QL Errors (Rolling Avg)')
sns.lineplot(y=moving_average(dqvs_fl_big,n=10000)[range(0,l,step)], x=range(0,len(dqvs_fl_big),step))
plt.xlabel('Iteration')
plt.ylabel('Change in Q (Error)')
ax.tick_params(axis='both', labelsize=9)
plt.savefig('QL Forest big error')