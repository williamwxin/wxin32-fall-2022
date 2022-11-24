import matplotlib.pyplot as plt
import numpy as np



def frozen_generator(p=0.125, width=16, fixer=False):
    np.random.seed(0)
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


fr = frozen_generator(width=50, fixer=False)
fr4 = frozen_generator(width=4)


conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr4)
p_non_holes = find_non_holes(fr4, 0.995)
P,R = conversion.P, conversion.R
P = fix_p(P)
pi = PolicyIteration(P,R,0.95)
ps_fl_small, deltas_fl_small = pi.run_non_holes(p_non_holes)
scatter_fl_small, labels_fl_small = ps_scatter_thresh(ps_fl_small, sort=True)

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

pi = PolicyIteration(P,R,0.995)
ps_fl_big_2, deltas_fl_big_2 = pi.run_non_holes(p_non_holes)
scatter_fl_big_2, labels_fl_big_2 = ps_scatter_thresh(ps_fl_big_2, x_step=1, y_step=1, thresh=0.95, sort=True)

pi = PolicyIteration(P,R,0.995)
pvs_fl_big_2, pps_fl_big_2, pvs_d_fl_big_2, pps_d_fl_big_2 = pi.run_non_holes_vp(p_non_holes)
pvs_scatter_fl_big_2, pps_scatter_fl_big_2, labels_fl_big_2 = vps_scatter_thresh(pvs_fl_big_2, pps_fl_big_2, x_step=10, y_step=1, thresh=0.95, sort=True)


conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr4)
p_non_holes = find_non_holes(fr4, 0.95)
P,R = conversion.P, conversion.R
P = fix_p(P)
pi = PolicyIteration(P,R,0.95)
pvs_fl_small_2, pps_fl_small_2,  pvs_d_fl_small_2, pps_d_fl_small_2 = pi.run_non_holes_vp(p_non_holes)
pvs_scatter_fl_small_2, pps_scatter_fl_small_2, labels_fl_small_2 = vps_scatter_thresh(pvs_fl_small_2, pps_fl_small_2, x_step=1, y_step=1, thresh=0.95, sort=True)


avg = 5

gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.995, 0.9995]
tvg_res_small = np.zeros((len(gammas),4))
tvg_res_small[:,0] = gammas
tvg_res_big = tvg_res_small.copy()

conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr4)
p_non_holes = find_non_holes(fr, 0.995)
P,R = conversion.P, conversion.R
P = fix_p(P)
pi = PolicyIteration(P,R,0.95)
ps_fl_small, deltas_fl_small = pi.run_non_holes(p_non_holes)

for i, g in enumerate(gammas):
    for a in range(avg):
        start = time.time()
        pi = ValueIteration(P, R, g)
        ps_i, deltas_i = pi.run()
        tvg_res_small[i,1] += (time.time() - start)/avg
        tvg_res_small[i,2] += ps_i[-1,0]/avg
        tvg_res_small[i,3] += len(ps_i)-1
    print(g)

conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr)
P,R = conversion.P, conversion.R
for i, g in enumerate(gammas):
    for a in range(avg):
        start = time.time()
        pi = ValueIteration(P, R, g, epsilon=0.001)
        ps_i, deltas_i = pi.run()
        tvg_res_big[i,1] += (time.time() - start)/avg
        tvg_res_big[i,2] += ps_i[-1,0]/avg
        tvg_res_big[i, 3] += len(ps_i) - 1
    print(g)

widths = [4, 8, 10, 15, 20, 25, 30,32,35,37,40,42,45,47,50]
tps_res = np.zeros((len(widths),3))
tps_res[:,0] = np.square(widths)

for i, w in enumerate(widths):
    for a in range(avg):
        fr_i = frozen_generator(width=w, fixer=True)
        conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr_i)
        p_non_holes = find_non_holes(fr_i, 0.995)
        P, R = conversion.P, conversion.R
        P = fix_p(P)

        start = time.time()
        pi = PolicyIteration(P, R, 0.995)
        ps_i, deltas_i = pi.run()
        tps_res[i,1] += (time.time() - start)/avg
        tps_res[i,2] += len(ps_i)-1
    print(w)

widths = [4, 8, 10, 15, 20, 25, 30,33,35,38,40,43,45,48,50,53,55]
tvs_fl_pivi = np.zeros((len(widths), 5))
tvs_fl_pivi[:, 0] = np.square(widths)

for i, s in enumerate(widths):
    for a in range(avg):
        fr_i = frozen_generator(width=w, fixer=True)
        conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr_i)
        p_non_holes = find_non_holes(fr_i, 0.995)
        P, R = conversion.P, conversion.R
        P = fix_p(P)

        start = time.time()
        vi = ValueIteration(P, R, 0.995, epsilon=0.001)
        vi.max_iter = 50000
        vs_i, deltas_i = vi.run()
        tvs_fl_pivi[i, 1] += (time.time() - start) / avg
        tvs_fl_pivi[i, 2] += (len(vs_i) - 1) / avg

        start = time.time()
        pi = PolicyIteration(P, R, 0.995)
        pi.max_iter = 50000
        ps_i, deltas_i = pi.run_non_holes(p_non_holes)
        tvs_fl_pivi[i, 3] += (time.time() - start) / avg
        tvs_fl_pivi[i, 4] += (len(ps_i) - 1) / avg
    print(s)



conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr)
P,R = conversion.P, conversion.R
vi = ValueIteration(P,R,0.995, epsilon=.001)
vi.verbose=True
vvs_fl_big_2, vps_fl_big_2, vvs_d_fl_big_2, vps_d_fl_big_2 = vi.run_vp()
vvs_scatter_fl_big_2, vps_scatter_fl_big_2, labels_fl_big_2 = vps_scatter_thresh(vvs_fl_big_2, vps_fl_big_2, x_step=25, y_step=1, thresh=1, sort=True)

conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr4)
P,R = conversion.P, conversion.R
vi = ValueIteration(P,R,0.95, epsilon=.001)
vi.verbose=True
vi.max_iter = 10000
vvs_fl_small_2, vps_fl_small_2, vvs_d_fl_small_2, vps_d_fl_small_2 = vi.run_vp()
vvs_scatter_fl_small_2, vps_scatter_fl_small_2, labels_fl_small_2 = vps_scatter_thresh(vvs_fl_small_2, vps_fl_small_2, x_step=1, y_step=1, thresh=0.95, sort=True)




sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-66)
plt.title('Frozen Lake 4x4 PI Values')
ax.scatter(scatter_fl_small[:,0], scatter_fl_small[:,1], scatter_fl_small[:,2], c=np.log((scatter_fl_small[:,2]*10+1)), cmap='jet')
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Value')
for x,y,z in labels_fl_small:
    ax.text(x,y,z,z, ha='center', fontsize=10)
ax.tick_params(axis='both', labelsize=9)
ax.text(0,0,max(scatter_fl_small[:,2])*1.5, 'Gamma=0.95', fontsize=11, color='purple')
plt.savefig('PI Frozen small values')

sns.set()
fig,ax = plt.subplots(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.2)
plt.title('Frozen Lake 4x4 PI Errors')
sns.lineplot(y=deltas_fl_small, x=range(1,len(deltas_fl_small)+1))
ax.tick_params(axis='x', labelsize=9)
plt.xlabel('Iteration')
plt.ylabel('# Policy Changes (Error)')
plt.savefig('PI Frozen small error')


sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-66)
plt.title('Frozen Lake 50x50 PI Values')
ax.scatter(scatter_fl_big[:,0], scatter_fl_big[:,1], scatter_fl_big[:,2], c=np.log((scatter_fl_big[:,2]*10+1)), cmap='jet')
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Value')
ax.tick_params(axis='both', labelsize=9)
for x,y,z in labels_fl_big:
    ax.text(x,y,z+.1,z, ha='center', fontsize=10)
ax.text(0,0,max(scatter_fl_big[:,2]), 'Gamma=0.95', fontsize=11, color='purple')
plt.savefig('PI Frozen big values')

sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-68)
plt.title('Frozen Lake 50x50 PI Values')
ax.scatter(scatter_fl_big_2[:,0], scatter_fl_big_2[:,1], scatter_fl_big_2[:,2], c=np.log((scatter_fl_big_2[:,2]*10+1)), cmap='jet')
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Value')
ax.tick_params(axis='both', labelsize=9)
for x,y,z in labels_fl_big_2:
    ax.text(x,y,z+.1,z, ha='center', fontsize=10)
ax.text(0,0,max(scatter_fl_big_2[:,2]), 'Gamma=0.995', fontsize=11, color='purple')
plt.savefig('PI Frozen big2 values')


colors=np.array(['blue', 'green', 'orange', 'red'])
colors_scatter = colors[pps_scatter_fl_big_2[:,2]]
sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-68)
plt.title('Frozen Lake 50x50 PI Policies')
ax.scatter(pvs_scatter_fl_big_2[:,0], pvs_scatter_fl_big_2[:,1], 0, c=colors_scatter)
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlim([0,1])
ax.tick_params(axis='both', labelsize=9)
ax.tick_params(axis='z', labelsize=0)
ax.grid(False)
ax.text(0,0,1.5, 'Blue=LEFT', fontsize=11, color='blue')
ax.text(0,0,1.3, 'Green=DOWN', fontsize=11, color='green')
ax.text(0,0,1.1, 'Orange=RIGHT', fontsize=11, color='orange')
ax.text(0,0,0.9, 'Red=UP', fontsize=11, color='red')
plt.savefig('PI Frozen big2 VP')


colors=np.array(['blue', 'green', 'orange', 'red'])
colors_scatter = colors[pps_scatter_fl_small_2[:,2]]
sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-68)
plt.title('Frozen Lake 4x4 PI Policies')
ax.scatter(pvs_scatter_fl_small_2[:,0], pvs_scatter_fl_small_2[:,1], 0, c=colors_scatter)
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlim([0,1])
ax.tick_params(axis='both', labelsize=9)
ax.tick_params(axis='z', labelsize=0)
ax.grid(False)
ax.text(0,0,1.5, 'Blue=LEFT', fontsize=11, color='blue')
ax.text(0,0,1.3, 'Green=DOWN', fontsize=11, color='green')
ax.text(0,0,1.1, 'Orange=RIGHT', fontsize=11, color='orange')
ax.text(0,0,0.9, 'Red=UP', fontsize=11, color='red')
plt.savefig('PI Frozen small VP')

sns.set()
fig,ax = plt.subplots(figsize=(2.75, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.26)
plt.title('Frozen 50x50 PI Errors')
sns.lineplot(y=deltas_fl_big_2, x=range(1,len(deltas_fl_big_2)+1))
plt.xlabel('Iteration')
plt.ylabel('# Policy Changes (Error)')
plt.savefig('PI Frozen big2 error')

# sns.set()
# fig, ax = plt.subplots(figsize=(3, 2))
# plt.subplots_adjust(bottom=.24)
# plt.subplots_adjust(left=.2)
# plt.title('4x4 Convg. Time vs Gamma')
# sns.lineplot(y=tvg_res_small[:,1]*1000, x=tvg_res_small[:,0], ax=ax)
# plt.vlines(x=0.4, ymin=0, ymax=max(tvg_res_small[:,1]*1000), color='purple', linestyles='dashed')
# plt.xlabel('Gamma')
# ax.text(0.5, 3, 'Threshold G=0.4', c='purple')
# plt.ylabel('Convg. Time (ms)')
# plt.savefig('pi Frozen small time v gamma')

sns.set()
fig, ax = plt.subplots(figsize=(2.5, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.25)
plt.title('4x4 Convg. Iter. vs Gamma')
sns.lineplot(y=tvg_res_small[:,3], x=tvg_res_small[:,0], ax=ax)
plt.vlines(x=0.4, ymin=0, ymax=max(tvg_res_small[:,3]), color='purple', linestyles='dashed')
plt.xlabel('Gamma')
ax.text(0.4, 200, 'Threshold G=0.4', c='purple')
plt.ylabel('Convg. Iterations')
plt.savefig('pi Frozen small iter v gamma')

#
# sns.set()
# fig, ax = plt.subplots(figsize=(2.5, 2.5))
# plt.subplots_adjust(bottom=.24)
# plt.subplots_adjust(left=.24)
# plt.title('50x50 Convg. Time vs Gamma')
# sns.lineplot(y=tvg_res_big[:,1]*1000, x=tvg_res_big[:,0], ax=ax)
# plt.vlines(x=0.995, ymin=0, ymax=max(tvg_res_big[:,1]*1000), color='purple', linestyles='dashed')
# ax.text(0.25, 3000, 'Threshold G=0.995', c='purple')
# plt.xlabel('Gamma')
# plt.ylabel('Convg. Time (ms)')
# plt.savefig('pi Frozen big time v gamma')

sns.set()
fig, ax = plt.subplots(figsize=(2.75, 2.75))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.25)
plt.title('50x50 Convg. Iter. vs Gamma')
sns.lineplot(y=tvg_res_big[:,3], x=tvg_res_big[:,0], ax=ax)
plt.vlines(x=0.995, ymin=0, ymax=max(tvg_res_big[:,3]), color='purple', linestyles='dashed')
ax.text(0.1, 2000, 'Threshold G=0.995', c='purple')
plt.xlabel('Gamma')
plt.ylabel('Convg. Iterations')
plt.savefig('pi Frozen big iter v gamma')

sns.set()
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax2 = ax.twinx()
ax.grid(False)
ax2.grid(False)
ax.set_ylabel('Time (sec)', color='blue')
ax2.set_ylabel('Iterations', color='green')
ax.tick_params(axis='y', labelsize=9.5, color = 'blue', labelcolor='blue')
ax2.tick_params(axis='y', labelsize=9.5, color = 'green', labelcolor='green')
plt.subplots_adjust(bottom=.22)
plt.subplots_adjust(left=.22)
plt.subplots_adjust(right=.74)
plt.title('G=0.995: Convg. vs State Size')
sns.lineplot(y=tps_res[:,1], x=tps_res[:,0], ax=ax, color='blue')
sns.lineplot(y=tps_res[:,2], x=tps_res[:,0], ax=ax2, color='green')
ax.set_xlabel('State Size')
plt.savefig('pi Frozen conv v states')

sns.set()
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.set_ylabel('Time per Iteration (ms)')
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.24)
plt.title('G=0.995: Time per Iteration')
sns.lineplot(y=tps_res[:,1]/tps_res[:,2]*1000, x=tps_res[:,0], ax=ax, color='blue')
ax.set_xlabel('State Size')
plt.savefig('pi Frozen conv2 v states')



sns.set()
fig,ax = plt.subplots(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.26)
plt.title('Frozen 4x4 VI vs PI: Policy Error')
sns.lineplot(y=vps_d_fl_small_2[1:], x=range(1,len(vps_d_fl_small_2)), label='Value Iter (VI)', linewidth=2)
sns.lineplot(y=pps_d_fl_small_2, x=range(1,len(pps_d_fl_small_2)+1), label='Policy Iter (PI)', linewidth=2)
ax.tick_params(axis='both', labelsize=9)
plt.xlabel('Iteration')
plt.ylabel('# Policy Changes (Error)')
plt.savefig('PI Frozen small PI vs VI')

sns.set()
fig,ax = plt.subplots(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.23)
plt.title('Frozen 50x50 VI vs PI: Policy Error')
sns.lineplot(y=vps_d_fl_big_2[1:], x=range(1,len(vps_d_fl_big_2)), label='Value Iter (VI)', linewidth=2)
sns.lineplot(y=pps_d_fl_big_2, x=range(1,len(pps_d_fl_big_2)+1), label='Policy Iter (PI)', linewidth=2)
ax.tick_params(axis='both', labelsize=9)
plt.xlabel('Iteration')
plt.ylabel('# Policy Changes (Error)')
plt.savefig('PI Frozen big PI vs VI')

sns.set()
fig,ax = plt.subplots(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.26)
plt.title('Frozen 4x4 VI vs PI: Value Error')
sns.lineplot(y=vvs_d_fl_small_2[1:], x=range(1,len(vvs_d_fl_small_2)), label='Value Iter (VI)', linewidth=2)
sns.lineplot(y=pvs_d_fl_small_2, x=range(1,len(pvs_d_fl_small_2)+1), label='Policy Iter (PI)', linewidth=2)
ax.tick_params(axis='both', labelsize=9)
plt.xlabel('Iteration')
plt.ylabel('Max Change in Value (Error)')
plt.savefig('PI Frozen small PI2 vs VI2')

sns.set()
fig,ax = plt.subplots(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.23)
plt.title('Frozen 50x50 VI vs PI: Value Error')
sns.lineplot(y=vvs_d_fl_big_2[1:], x=range(1,len(vvs_d_fl_big_2)), label='Value Iter (VI)', linewidth=2)
sns.lineplot(y=pvs_d_fl_big_2, x=range(1,len(pvs_d_fl_big_2)+1), label='Policy Iter (PI)', linewidth=2)
ax.tick_params(axis='both', labelsize=9)
plt.xlabel('Iteration')
plt.ylabel('Max Change in Value (Error)')
plt.savefig('PI Frozen big PI2 vs VI2')



sns.set()
fig,ax = plt.subplots(figsize=(2.25, 2.25))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.26)
plt.title('Frozen Time vs # States')
sns.lineplot(y=tvs_fl_pivi[:,1], x=tvs_fl_pivi[:,0], label='VI', linewidth=2)
sns.lineplot(y=tvs_fl_pivi[:,3], x=tvs_fl_pivi[:,0], label='PI', linewidth=2)
ax.tick_params(axis='both', labelsize=9)
ax.text(0,1.5,'G=0.995',fontsize=10, color='purple')
plt.xlabel('State Size')
plt.ylabel('Time (sec)')
plt.savefig('PI Frozen pivi TVS')

sns.set()
fig,ax = plt.subplots(figsize=(2.25, 2.25))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.25)
plt.title('Frozen Time per Iter')
sns.lineplot(y=tvs_fl_pivi[:,1]/tvs_fl_pivi[:,2]*1000, x=tvs_fl_pivi[:,0], label='VI', linewidth=2)
sns.lineplot(y=tvs_fl_pivi[:,3]/tvs_fl_pivi[:,4]*1000, x=tvs_fl_pivi[:,0], label='PI', linewidth=2)
ax.tick_params(axis='both', labelsize=9)
plt.xlabel('State Size')
plt.ylabel('Time per Iter (ms)')
plt.savefig('PI Frozen pivi TVS per iteration')

sns.set()
fig,ax = plt.subplots(figsize=(2.25, 2.25))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.26)
plt.title('Frozen # Iter vs # States')
sns.lineplot(y=tvs_fl_pivi[:,2], x=tvs_fl_pivi[:,0], label='VI', linewidth=2)
sns.lineplot(y=tvs_fl_pivi[:,4], x=tvs_fl_pivi[:,0], label='PI', linewidth=2)
ax.tick_params(axis='both', labelsize=9)
plt.xlabel('State Size')
plt.ylabel('Iterations')
plt.savefig('PI Frozen pivi TVS # iterations')

