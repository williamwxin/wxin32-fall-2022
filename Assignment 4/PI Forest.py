import matplotlib.pyplot as plt
import numpy as np
import hiive.mdptoolbox as mdp



def frozen_generator(p=0.125, width=16, fixer=False):
    arr = ['S']
    for i in range(width*width-2):
        if random.random() >= p:
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




P,R = mdp.example.forest(16, p=0.2)
pi = PolicyIteration(P,R,0.95)
ps_fo_small, deltas_fo_small = pi.run()
scatter_fo_small, labels_fo_small = ps_scatter_thresh(ps_fo_small, sort=True)

P,R = mdp.example.forest(2500, p=0.2)
pi = PolicyIteration(P,R,0.95)
pi.verbose=True
ps_fo_big, deltas_fo_big = pi.run()
scatter_fo_big, labels_fo_big = ps_scatter_thresh(ps_fo_big, x_step=1, y_step=1, thresh=.9, sort=True)

P,R = mdp.example.forest(2500, p=0.0001)
pi = PolicyIteration(P,R,0.9995)
pi.verbose=True
pi.max_iter = 3000
ps_fo_big_2, deltas_fo_big_2 = pi.run()
scatter_fo_big_2, labels_fo_big_2 = ps_scatter_thresh(ps_fo_big_2, x_step=10, y_step=50, thresh=0.95, sort=True)


P,R = mdp.example.forest(2500, p=0.0001)
pi = PolicyIteration(P,R,0.9995)
pi.verbose=True
pi.max_iter = 3000
pvs_fo_big_2, pps_fo_big_2, deltas_fo_big_2,_ = pi.run_vp()
pvs_scatter_fo_big_2, pps_scatter_fo_big_2, labels_fo_big_2 = vps_scatter_thresh(pvs_fo_big_2, pps_fo_big_2, x_step=25, y_step=50, thresh=1, sort=True)

P,R = mdp.example.forest(16, p=0.2)
pi = PolicyIteration(P,R,0.95)
pi.verbose=True
pvs_fo_small_2, pps_fo_small_2, deltas_fo_small_2,_ = pi.run_vp()
pvs_scatter_fo_small_2, pps_scatter_fo_small_2, labels_fo_small_2 = vps_scatter_thresh(pvs_fo_small_2, pps_fo_small_2, x_step=1, y_step=1, thresh=0.95, sort=True)

P,R = mdp.example.forest(2500, p=0.0001)
vi = ValueIteration(P,R,0.9995, epsilon=.001)
vi.verbose=True
vvs_fo_big_2, vps_fo_big_2, vvs_d_fo_big_2, vps_d_fo_big_2 = vi.run_vp()
vvs_scatter_fo_big_2, vps_scatter_fo_big_2, labels_fo_big_2 = vps_scatter_thresh(vvs_fo_big_2, vps_fo_big_2, x_step=25, y_step=50, thresh=1, sort=True)

P,R = mdp.example.forest(16, p=0.2)
vi = ValueIteration(P,R,0.95, epsilon=.001)
vi.verbose=True
vi.max_iter = 10000
vvs_fo_small_2, vps_fo_small_2, vvs_d_fo_small_2, vps_d_fo_small_2 = vi.run_vp()
vvs_scatter_fo_small_2, vps_scatter_fo_small_2, labels_fo_small_2 = vps_scatter_thresh(vvs_fo_small_2, vps_fo_small_2, x_step=1, y_step=1, thresh=0.95, sort=True)

P,R = mdp.example.forest(2500, p=0.0001)
pi = PolicyIteration(P,R,0.9995)
pi.max_iter = 3000
pi.verbose = True
pvs_fo_big_2, pps_fo_big_2, pvs_d_fo_big_2, pps_d_fo_big_2 = pi.run_vp()
pvs_scatter_fo_big_2, pps_scatter_fo_big_2, labels_fo_big_2 = vps_scatter_thresh(pvs_fo_big_2, pps_fo_big_2, x_step=10, y_step=1, thresh=0.95, sort=True)


P,R = mdp.example.forest(16, p=0.2)
pi = PolicyIteration(P,R,0.95)
pvs_fo_small_2, pps_fo_small_2,  pvs_d_fo_small_2, pps_d_fo_small_2 = pi.run_vp()
pvs_scatter_fo_small_2, pps_scatter_fo_small_2, labels_fo_small_2 = vps_scatter_thresh(pvs_fo_small_2, pps_fo_small_2, x_step=1, y_step=1, thresh=0.95, sort=True)



avg = 5

gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.995, 0.9995]
tvg_res_small = np.zeros((len(gammas),4))
tvg_res_small[:,0] = gammas
tvg_res_big = tvg_res_small.copy()


widths = [4, 8, 10, 15, 20, 25, 30, 33, 35, 38, 40, 43, 45, 48, 50, 53, 55]
tvs_fl_pivi = np.zeros((len(widths), 5))
tvs_fl_pivi[:, 0] = np.square(widths)

for i, w in enumerate(widths):
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
    print(w)


sizes = [5,10,25,50,100,250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000]
tvs_fo_pivi = np.zeros((len(sizes),5))
tvs_fo_pivi[:,0] = sizes
avg = 2
for i, s in enumerate(sizes):
    for a in range(avg):
        P,R = mdp.example.forest(s,p=0.0001)
        start = time.time()
        vi = ValueIteration(P, R, 0.9995, epsilon=0.001)
        vi.max_iter = 50000
        vs_i, deltas_i = vi.run()
        tvs_fo_pivi[i,1] += (time.time() - start)/avg
        tvs_fo_pivi[i,2] += (len(vs_i)-1)/avg

        P, R = mdp.example.forest(s, p=0.0001)
        start = time.time()
        pi = PolicyIteration(P, R, 0.9995)
        pi.max_iter = 50000
        ps_i, deltas_i = pi.run()
        tvs_fo_pivi[i, 3] += (time.time() - start) / avg
        tvs_fo_pivi[i, 4] += (len(ps_i) - 1) / avg
    print(s)


widths = [4, 8, 10, 15, 20, 25, 30, 33, 35, 38, 40, 43, 45, 48, 50, 53, 55]
tvs_fl_pivi = np.zeros((len(widths), 5))
tvs_fl_pivi[:, 0] = np.square(widths)

for i, w in enumerate(widths):
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
    print(w)


gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.995, 0.9995]
tvg_small_fo_pivi = np.zeros((len(gammas),5))
tvg_small_fo_pivi[:,0] = gammas
tvg_big_fo_pivi = tvg_small_fo_pivi.copy()
avg = 3
for i, g in enumerate(gammas):
    for a in range(avg):
        P,R = mdp.example.forest(16,p=0.0001)
        start = time.time()
        vi = ValueIteration(P, R, g, epsilon=0.001)
        vi.max_iter = 50000
        vs_i, deltas_i = vi.run()
        tvg_small_fo_pivi[i,1] += (time.time() - start)/avg
        tvg_small_fo_pivi[i,2] += (len(vs_i)-1)/avg

        P, R = mdp.example.forest(16, p=0.0001)
        start = time.time()
        pi = PolicyIteration(P, R, g)
        pi.max_iter = 50000
        ps_i, deltas_i = pi.run()
        tvg_small_fo_pivi[i, 3] += (time.time() - start) / avg
        tvg_small_fo_pivi[i, 4] += (len(ps_i)) / avg

        P, R = mdp.example.forest(2500, p=0.0001)
        start = time.time()
        vi = ValueIteration(P, R, g, epsilon=0.001)
        vi.max_iter = 50000
        vs_i, deltas_i = vi.run()
        tvg_big_fo_pivi[i, 1] += (time.time() - start) / avg
        tvg_big_fo_pivi[i, 2] += (len(vs_i) - 1) / avg

        P, R = mdp.example.forest(2500, p=0.0001)
        start = time.time()
        pi = PolicyIteration(P, R, g)
        pi.max_iter = 50000
        ps_i, deltas_i = pi.run()
        tvg_big_fo_pivi[i, 3] += (time.time() - start) / avg
        tvg_big_fo_pivi[i, 4] += (len(ps_i)) / avg
    print(g)


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
        P, R = conversion.P, conversion.R
        start = time.time()
        pi = PolicyIteration(P, R, 0.995)
        ps_i, deltas_i = pi.run()
        tps_res[i,1] += (time.time() - start)/avg
        tps_res[i,2] += len(ps_i)-1
    print(w)


P,R = mdp.example.forest(16, p=0.2)
vi = ValueIteration(P,R,0.95, epsilon=0.001)
vips_fo_small, deltas_fo_small = vi.run_p()
vs_fo_small = np.array(vs_fo_small)
scatter_fo_small, labels_fo_small = vs_scatter_thresh(vs_fo_small, sort=False)


sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-66)
plt.title('Forest 16 PI Values')
ax.scatter(scatter_fo_small[:,0], scatter_fo_small[:,1], scatter_fo_small[:,2], c=np.log((scatter_fo_small[:,2]*10+1)), cmap='jet')
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Value')
for x,y,z in labels_fo_small:
    ax.text(x,y,z,z, ha='center', fontsize=10)
ax.tick_params(axis='both', labelsize=9)
ax.text(0,0,max(scatter_fo_small[:,2])*1.3, 'Gamma=0.95', fontsize=11, color='purple')
ax.text(0,0,max(scatter_fo_small[:,2])*1.2, 'P=0.2', fontsize=11, color='purple')
plt.savefig('PI Forest small values')

sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.23)
plt.title('Forest 16 PI Errors')
sns.lineplot(y=deltas_fo_small, x=range(1,len(deltas_fo_small)+1))
plt.xlabel('Iteration')
plt.ylabel('# Policy Changes (Error)')
plt.savefig('PI Forest small error')


sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-66)
plt.title('Forest 2500 PI Values')
ax.scatter(scatter_fo_big[:,0], scatter_fo_big[:,1], scatter_fo_big[:,2], c=np.log((scatter_fo_big[:,2]*10+1)), cmap='jet')
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Value')
ax.tick_params(axis='both', labelsize=9)
for x,y,z in labels_fo_big:
    ax.text(x,y,z+.1,z, ha='center', fontsize=10)
ax.text(0,0,max(scatter_fo_big[:,2]), 'Gamma=0.95', fontsize=11, color='purple')
plt.savefig('PI Forest big values')

sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-68)
plt.title('Forest 2500 PI Values')
ax.scatter(scatter_fo_big_2[:,0], scatter_fo_big_2[:,1], scatter_fo_big_2[:,2], c=np.log((scatter_fo_big_2[:,2]*10+1)), cmap='jet')
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Value')
ax.tick_params(axis='both', labelsize=9)
for x,y,z in labels_fo_big_2:
    ax.text(x,y,z+.1,np.round(z,1), ha='center', fontsize=10)
ax.text(0,0,max(scatter_fo_big_2[:,2])*1.3, 'Gamma=0.9995', fontsize=11, color='purple')
ax.text(0,0,max(scatter_fo_big_2[:,2])*1.1, 'P=0.0001', fontsize=11, color='purple')
plt.savefig('PI Forest big2 values')

sns.set()
fig,ax = plt.subplots(figsize=(2.75, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.26)
plt.title('Forest 2500 PI Errors')
sns.lineplot(y=deltas_fo_big_2, x=range(1,len(deltas_fo_big_2)+1))
plt.xlabel('Iteration')
plt.ylabel('# Policy Changes (Error)')
plt.savefig('PI Forest big2 error')

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



colors=np.array(['green', 'purple'])
colors_scatter = colors[pps_scatter_fo_big_2[:,2]]
sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-68)
plt.title('Forest 2500 PI Policies')
ax.scatter(pvs_scatter_fo_big_2[:,0], pvs_scatter_fo_big_2[:,1], 0, c=colors_scatter)
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlim([0,1])
ax.tick_params(axis='both', labelsize=9)
ax.tick_params(axis='z', labelsize=0)
ax.grid(False)
ax.text(0,0,1.5, 'Green=WAIT', fontsize=11, color='green')
ax.text(0,0,1.3, 'Purple=CUT', fontsize=11, color='purple')
plt.savefig('PI Forest big2 VP')


colors=np.array(['green', 'purple'])
colors_scatter = colors[pps_scatter_fo_small_2[:,2]]
sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-68)
plt.title('Forest 16 PI Policies')
ax.scatter(pvs_scatter_fo_small_2[:,0], pvs_scatter_fo_small_2[:,1], 0, c=colors_scatter)
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlim([0,1])
ax.tick_params(axis='both', labelsize=9)
ax.tick_params(axis='z', labelsize=0)
ax.grid(False)
ax.text(0,0,1.5, 'Green=WAIT', fontsize=11, color='green')
ax.text(0,0,1.3, 'Purple=CUT', fontsize=11, color='purple')
plt.savefig('PI Forest small VP')


sns.set()
fig,ax = plt.subplots(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.26)
plt.title('Forest 16 VI vs PI: Policy Error')
sns.lineplot(y=vps_d_fo_small_2[1:], x=range(1,len(vps_d_fo_small_2)), label='Value Iter (VI)')
sns.lineplot(y=pps_d_fo_small_2, x=range(1,len(pps_d_fo_small_2)+1), label='Policy Iter (PI)', linewidth=2)
ax.tick_params(axis='both', labelsize=9)
plt.xlabel('Iteration')
plt.ylabel('# Policy Changes (Error)')
plt.savefig('PI Forest small PI vs VI')

sns.set()
fig,ax = plt.subplots(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.23)
plt.title('Forest 2500 VI vs PI: Policy Error')
sns.lineplot(y=vps_d_fo_big_2[1:], x=range(1,len(vps_d_fo_big_2)), label='Value Iter (VI)')
sns.lineplot(y=pps_d_fo_big_2, x=range(1,len(pps_d_fo_big_2)+1), label='Policy Iter (PI)', linewidth=2)
ax.tick_params(axis='both', labelsize=9)
plt.xlabel('Iteration')
plt.ylabel('# Policy Changes (Error)')
plt.savefig('PI Forest big PI vs VI')


sns.set()
fig,ax = plt.subplots(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.26)
plt.title('Forest 16 VI vs PI: Value Error')
sns.lineplot(y=vvs_d_fo_small_2[1:], x=range(1,len(vvs_d_fo_small_2)), label='Value Iter (VI)', linewidth=2)
sns.lineplot(y=pvs_d_fo_small_2, x=range(1,len(pvs_d_fo_small_2)+1), label='Policy Iter (PI)', linewidth=2)
ax.tick_params(axis='both', labelsize=9)
plt.xlabel('Iteration')
plt.ylabel('# Policy Changes (Error)')
plt.savefig('PI Forest small PI2 vs VI2')

sns.set()
fig,ax = plt.subplots(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.23)
plt.title('Forest 2500 VI vs PI: Value Error')
sns.lineplot(y=vvs_d_fo_big_2[1:], x=range(1,len(vvs_d_fo_big_2)), label='Value Iter (VI)', linewidth=2)
sns.lineplot(y=pvs_d_fo_big_2, x=range(1,len(pvs_d_fo_big_2)+1), label='Policy Iter (PI)', linewidth=2)
ax.tick_params(axis='both', labelsize=9)
ax.text(0,200,'4.00',fontsize=10, color='blue')
ax.text(5000,200,round(vvs_d_fo_big_2[5000],2),fontsize=10, color='blue')
plt.xlabel('Iteration')
plt.ylabel('# Policy Changes (Error)')
plt.savefig('PI Forest big PI2 vs VI2')




sns.set()
fig,ax = plt.subplots(figsize=(2.25, 2.25))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.26)
plt.title('Forest Time vs # States')
sns.lineplot(y=tvs_fo_pivi[:,1], x=tvs_fo_pivi[:,0], label='VI', linewidth=2)
sns.lineplot(y=tvs_fo_pivi[:,3], x=tvs_fo_pivi[:,0], label='PI', linewidth=2)
ax.tick_params(axis='both', labelsize=9)
ax.text(0,250,'G=0.9995',fontsize=10, color='purple')
ax.text(0,150,'P=0.0001',fontsize=10, color='purple')
plt.xlabel('State Size')
plt.ylabel('Time (sec)')
plt.savefig('PI Forest pivi TVS')

sns.set()
fig,ax = plt.subplots(figsize=(2.25, 2.25))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.25)
plt.title('Forest Time per Iter')
sns.lineplot(y=tvs_fo_pivi[:,1]/tvs_fo_pivi[:,2]*1000, x=tvs_fo_pivi[:,0], label='VI', linewidth=2)
sns.lineplot(y=tvs_fo_pivi[:,3]/tvs_fo_pivi[:,4]*1000, x=tvs_fo_pivi[:,0], label='PI', linewidth=2)
ax.tick_params(axis='both', labelsize=9)
plt.xlabel('State Size')
plt.ylabel('Time per Iter (ms)')
plt.savefig('PI Forest pivi TVS per iteration')


sns.set()
fig,ax = plt.subplots(figsize=(2.25, 2.25))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.3)
plt.title('Forest # Iter vs # States')
sns.lineplot(y=tvs_fo_pivi[:,2], x=tvs_fo_pivi[:,0], label='VI', linewidth=2)
sns.lineplot(y=tvs_fo_pivi[:,4], x=tvs_fo_pivi[:,0], label='PI', linewidth=2)
ax.tick_params(axis='both', labelsize=9)
plt.xlabel('State Size')
plt.ylabel('Iterations')
plt.savefig('PI Forest pivi TVS # iterations')
