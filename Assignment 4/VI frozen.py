import matplotlib.pyplot as plt
import numpy as np



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

def v_non_holes(fr):
    fr_flat = fr.flatten()
    fr_non_holes = np.array(range(len(fr_flat)))[fr_flat!='H'][0:-1]
    return fr_non_holes

fr = frozen_generator(width=50, fixer=True)
fr4 = frozen_generator(width=4)


conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr4)
P,R = conversion.P, conversion.R
vi = ValueIteration(P,R,0.95, epsilon=0.001)
vs_fl_small, deltas_fl_small = vi.run()
vs_fl_small = np.array(vs_fl_small)
scatter_fl_small, labels_fl_small = vs_scatter_thresh(vs_fl_small[:,v_non_holes(fr4)], sort=True)


conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr)
P,R = conversion.P, conversion.R
vi = ValueIteration(P,R,0.95, epsilon=0.001)
vs_fl_big, deltas_fl_big = vi.run()
scatter_fl_big, labels_fl_big = vs_scatter_thresh(vs_fl_big[:,v_non_holes(fr)], x_step=100, y_step=2, thresh=0.95, sort=True)

vi = ValueIteration(P,R,0.995, epsilon=0.0005)
vi.verbose=True
vs_fl_big_2, deltas_fl_big_2 = vi.run()
scatter_fl_big_2, labels_fl_big_2 = vs_scatter_thresh(vs_fl_big_2[:,v_non_holes(fr)], x_step=100, y_step=10, thresh=0.95, sort=True)

avg = 5

gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.995, 0.9995]
tvg_res_small = np.zeros((len(gammas),4))
tvg_res_small[:,0] = gammas
tvg_res_big = tvg_res_small.copy()

conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr4)
P,R = conversion.P, conversion.R
for i, g in enumerate(gammas):
    for a in range(avg):
        start = time.time()
        vi = ValueIteration(P, R, g, epsilon=0.001)
        vs_i, deltas_i = vi.run()
        tvg_res_small[i,1] += (time.time() - start)/avg
        tvg_res_small[i,2] += vs_i[-1,0]/avg
        tvg_res_small[i,3] += len(vs_i)-1
    print(g)

conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr)
P,R = conversion.P, conversion.R
for i, g in enumerate(gammas):
    for a in range(avg):
        start = time.time()
        vi = ValueIteration(P, R, g, epsilon=0.001)
        vs_i, deltas_i = vi.run()
        tvg_res_big[i,1] += (time.time() - start)/avg
        tvg_res_big[i,2] += vs_i[-1,0]/avg
        tvg_res_big[i, 3] += len(vs_i) - 1
    print(g)

widths = [4, 8, 10, 15, 20, 25, 30,32,35,37,40,42,45,47,50]
fl_tvs_res = np.zeros((len(widths),3))
fl_tvs_res[:,0] = np.square(widths)
avg = 3
for i, w in enumerate(widths):
    for a in range(avg):
        fr_i = frozen_generator(width=w, fixer=True)
        conversion = OpenAI_MDPToolbox("FrozenLake-v1", desc=fr_i)
        P, R = conversion.P, conversion.R
        start = time.time()
        vi = ValueIteration(P, R, 0.995, epsilon=0.001)
        vs_i, deltas_i = vi.run()
        fl_tvs_res[i,1] += (time.time() - start)/avg
        fl_tvs_res[i,2] += (len(vs_i)-1)/avg
    print(w)


sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-66)
plt.title('Frozen Lake 4x4 VI Values')
ax.scatter(scatter_fl_small[:,0], scatter_fl_small[:,1], scatter_fl_small[:,2], c=np.log((scatter_fl_small[:,2]*10+1)), cmap='jet')
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Value')
for x,y,z in labels_fl_small:
    ax.text(x,y,z,z, ha='center', fontsize=10)
ax.tick_params(axis='both', labelsize=9)
ax.text(0,0,max(scatter_fl_small[:,2])*1.5, 'Gamma=0.95', fontsize=11, color='purple')
plt.savefig('VI Frozen small values')

sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.2)
plt.title('Frozen Lake 4x4 VI Errors')
sns.lineplot(y=deltas_fl_small, x=range(len(deltas_fl_small)))
plt.xlabel('Iteration')
plt.ylabel('Max Change in Value (Error)')
plt.savefig('VI Frozen small error')


sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-66)
plt.title('Frozen Lake 50x50 VI Values')
ax.scatter(scatter_fl_big[:,0], scatter_fl_big[:,1], scatter_fl_big[:,2], c=np.log((scatter_fl_big[:,2]*10+1)), cmap='jet')
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Value')
ax.tick_params(axis='both', labelsize=9)
for x,y,z in labels_fl_big:
    ax.text(x,y,z,z, ha='center', fontsize=10)
ax.text(0,0,max(scatter_fl_big[:,2]), 'Gamma=0.95', fontsize=11, color='purple')
plt.savefig('VI Frozen big values')

sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-68)
plt.title('Frozen Lake 50x50 VI Values')
ax.scatter(scatter_fl_big_2[:,0], scatter_fl_big_2[:,1], scatter_fl_big_2[:,2], c=np.log((scatter_fl_big_2[:,2]*10+1)), cmap='jet')
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Value')
ax.tick_params(axis='both', labelsize=9)
for x,y,z in labels_fl_big_2:
    ax.text(x,y,z,z, ha='center', fontsize=10)
ax.text(0,0,max(scatter_fl_big_2[:,2]), 'Gamma=0.995', fontsize=11, color='purple')
plt.savefig('VI Frozen big2 values')

sns.set()
fig,ax = plt.subplots(figsize=(2.75, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.21)
plt.title('50x50 VI Errors')
sns.lineplot(y=deltas_fl_big_2, x=range(len(deltas_fl_big_2)))
plt.xlabel('Iteration')
plt.ylabel('Max Change in Value (Error)')
ax.text(100,0.3, 'Gamma=0.995', fontsize=11, color='purple')
plt.savefig('VI Frozen big2 error')

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
# plt.savefig('VI Frozen small time v gamma')

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
plt.savefig('VI Frozen small iter v gamma')



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
# plt.savefig('VI Frozen big time v gamma')

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
plt.savefig('VI Frozen big iter v gamma')


sns.set()
fig, ax = plt.subplots(figsize=(2.75, 2.75))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.25)
plt.title('50x50 Convg. Iter. vs Gamma')
sns.lineplot(y=tvg_res_big[:,1]*1000, x=1/(1-tvg_res_big[:,0])*np.log(1/(1-tvg_res_big[:,0])), ax=ax)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Gamma')
plt.ylabel('Convg. Iterations')
plt.savefig('VI Frozen big iter v gamma')
=

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
plt.title('Frozen Convg. vs State Size')
sns.lineplot(y=fl_tvs_res[:,1], x=fl_tvs_res[:,0], ax=ax, color='blue')
sns.lineplot(y=fl_tvs_res[:,2], x=fl_tvs_res[:,0], ax=ax2, color='green')
ax.set_xlabel('State Size')
ax.tick_params(axis='x', labelsize=9.5)
ax.text(600, 2, 'G=0.995', c='purple')
plt.savefig('VI Frozen conv v states')

sns.set()
fig, ax = plt.subplots(figsize=(2.5, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.24)
plt.title('Frozen Time per Iteration')
ax.set_ylabel('Time per Iteration (ms)', color='blue')
sns.lineplot(y=fl_tvs_res[:,1]/fl_tvs_res[:,2]*1000, x=fl_tvs_res[:,0], ax=ax, color='blue')
ax.set_xlabel('State Size')
ax.tick_params(axis='y', labelsize=9.5, color = 'blue', labelcolor='blue')
plt.savefig('VI Frozen conv2 v states')