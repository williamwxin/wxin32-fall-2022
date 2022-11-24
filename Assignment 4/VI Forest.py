import matplotlib.pyplot as plt
import numpy as np
import hiive.mdptoolbox as mdp



def vs_scatter_thresh(vs, x_step=1, y_step=1, thresh = 1, sort=True):
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




P,R = mdp.example.forest(16, p=0.2)
vi = ValueIteration(P,R,0.95, epsilon=0.001)
vi.max_iter = 1000
vs_fo_small, deltas_fo_small = vi.run()
vs_fo_small = np.array(vs_fo_small)
scatter_fo_small, labels_fo_small = vs_scatter_thresh(vs_fo_small, sort=False)


P,R = mdp.example.forest(2500, p=0.2)
vi = ValueIteration(P,R,0.95, epsilon=0.001)
vi.max_iter = 1000
vs_fo_big, deltas_fo_big = vi.run()
scatter_fo_big, labels_fo_big = vs_scatter_thresh(vs_fo_big, x_step=100, y_step=2, thresh=0.95, sort=True)

P,R = mdp.example.forest(2500, p=0.0001)
vi = ValueIteration(P,R,0.9995, epsilon=0.001)
vs_fo_big_2, deltas_fo_big_2 = vi.run()
scatter_fo_big_2, labels_fo_big_2 = vs_scatter_thresh(vs_fo_big_2, x_step=100, y_step=200, thresh=0.95, sort=True)

avg = 5

gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.995, 0.9995]
fo_tvg_res_small = np.zeros((len(gammas),4))
fo_tvg_res_small[:,0] = gammas
fo_tvg_res_big = fo_tvg_res_small.copy()

P,R = mdp.example.forest(16)
for i, g in enumerate(gammas):
    for a in range(avg):
        start = time.time()
        vi = ValueIteration(P, R, g, epsilon=0.001)
        vs_i, deltas_i = vi.run()
        fo_tvg_res_small[i,1] += (time.time() - start)/avg
        fo_tvg_res_small[i,2] += vs_i[-1,0]/avg
        fo_tvg_res_small[i,3] += len(vs_i)-1
    print(g)

P,R = mdp.example.forest(2500)
for i, g in enumerate(gammas):
    for a in range(avg):
        start = time.time()
        vi = ValueIteration(P, R, g, epsilon=0.001)
        vs_i, deltas_i = vi.run()
        fo_tvg_res_big[i,1] += (time.time() - start)/avg
        fo_tvg_res_big[i,2] += vs_i[-1,0]/avg
        fo_tvg_res_big[i, 3] += len(vs_i) - 1
    print(g)

sizes = [5,10,25,50,100,250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000]
tvs_res = np.zeros((len(sizes),3))
tvs_res[:,0] = sizes
avg = 2

for i, s in enumerate(sizes):
    for a in range(avg):
        P,R = mdp.example.forest(s,p=0.0001)
        start = time.time()
        vi = ValueIteration(P, R, 0.9995, epsilon=0.001)
        vs_i, deltas_i = vi.run()
        tvs_res[i,1] += (time.time() - start)/avg
        tvs_res[i,2] += (len(vs_i)-1)/avg
    print(s)

gs = np.arange(0.8,1,0.01)
ps = np.arange(0,0.2,0.01)

gammap_time = []
gammap_scatter = []
for x, g in enumerate(gs):
    for y, p in enumerate(ps):
        for a in range(avg*2):
            start = time.time()
            P, R = mdp.example.forest(16, p=p)
            vi = ValueIteration(P, R, g, epsilon=0.001)
            vs_fo_big, deltas_fo_big = vi.run()
        policy = vi.policy[1]
        gammap_time.append([g,p,(time.time()-start)/avg/2*1000])
        gammap_scatter.append([g,p,policy])
gammap_scatter = np.array(gammap_scatter)
gammap_time = np.array(gammap_time)




sns.set()
fig,ax = plt.subplots(figsize=(3.5, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.2)
plt.subplots_adjust(right=.78)
plt.title('Forest: State 1 Policy')
ax.scatter(gammap_scatter[:,0], gammap_scatter[:,1], c=gammap_scatter[:,2], cmap='jet')
ax.set_xlabel('Gamma')
ax.set_ylabel('P (Prob of Forest Burn)')
ax.tick_params(axis='both', labelsize=9)
ax.text(1,0.15, 'Red=Cut', fontsize=11, color='red')
ax.text(1,0.05, 'Blue=Wait', fontsize=11, color='blue')
plt.savefig('VI Forest gammap')

sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(8,-139)
ax.scatter(gammap_time[:,0], gammap_time[:,1], gammap_time[:,2], c=gammap_scatter[:,2], cmap='jet')
ax.set_zlabel('Run Time (ms)')
ax.tick_params(axis='both', labelsize=0)
ax.tick_params(axis='z', labelsize=9)
plt.savefig('VI Forest gammap time')

sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-66)
plt.title('Forest 16 VI Values')
ax.scatter(scatter_fo_small[:,0], scatter_fo_small[:,1], scatter_fo_small[:,2], c=np.log(scatter_fo_small[:,2]/500+1), cmap='jet')
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Value')
for x,y,z in labels_fo_small:
    ax.text(x,y,z+1,z, ha='center', fontsize=10)
ax.tick_params(axis='both', labelsize=9)
ax.text(0,0,max(scatter_fo_small[:,2])*1.5, 'Gamma=0.95', fontsize=11, color='purple')
plt.savefig('VI Forest small values')

sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.2)
plt.title('Forest 16 VI Errors')
sns.lineplot(y=deltas_fo_small, x=range(len(deltas_fo_small)))
plt.xlabel('Iteration')
plt.ylabel('Max Change in Value (Error)')
plt.savefig('VI Forest small error')


sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-66)
plt.title('Forest 2500 VI Values')
ax.scatter(scatter_fo_big[:,0], scatter_fo_big[:,1], scatter_fo_big[:,2], c=np.log((scatter_fo_big[:,2]/500+1)), cmap='jet')
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Value')
ax.tick_params(axis='both', labelsize=9)
for x,y,z in np.round(labels_fo_big,1):
    ax.text(x,y,z+1,z, ha='center', fontsize=10)
ax.text(0,0,max(scatter_fo_big[:,2])*1.6, 'Gamma=0.95', fontsize=11, color='purple')
ax.text(0,0,max(scatter_fo_big[:,2])*1.4, 'P=0.2', fontsize=11, color='purple')
plt.savefig('VI Forest big values')

sns.set()
fig = plt.figure(figsize=(3, 2.5))
plt.subplots_adjust(bottom=.125)
ax = fig.add_subplot(projection='3d')
ax.view_init(23,-68)
plt.title('Forest 2500 VI Values')
ax.scatter(scatter_fo_big_2[:,0], scatter_fo_big_2[:,1], scatter_fo_big_2[:,2], c=np.log((scatter_fo_big_2[:,2]/1000+1)), cmap='jet')
ax.set_xlabel('State')
ax.set_ylabel('Iteration')
ax.set_zlabel('Value')
ax.tick_params(axis='both', labelsize=9)
for x,y,z in np.round(labels_fo_big_2,1):
    ax.text(x,y,z,z, ha='center', fontsize=10)
ax.text(0,0,max(scatter_fo_big_2[:,2])*1.6, 'Gamma=0.9995', fontsize=11, color='purple')
ax.text(0,0,max(scatter_fo_big_2[:,2])*1.4, 'P=0.0001', fontsize=11, color='purple')
plt.savefig('VI Forest big2 values')

sns.set()
fig,ax = plt.subplots(figsize=(2.75, 2.5))
plt.subplots_adjust(bottom=.2)
plt.subplots_adjust(left=.21)
plt.title('2500 VI Errors')
sns.lineplot(y=deltas_fo_big_2, x=range(len(deltas_fo_big_2)))
plt.xlabel('Iteration')
plt.ylabel('Max Change in Value (Error)')
ax.text(5000,max(deltas_fo_big_2)*0.9, 'Gamma=0.9995', fontsize=11, color='purple')
ax.text(5000,max(deltas_fo_big_2)*0.8, 'P=0.0001', fontsize=11, color='purple')
plt.savefig('VI Forest big2 error')

sns.set()
fig, ax = plt.subplots(figsize=(3, 2))
plt.subplots_adjust(bottom=.24)
plt.subplots_adjust(left=.2)
plt.title('16 Convg. Time vs Gamma')
sns.lineplot(y=fo_tvg_res_small[:,1]*1000, x=fo_tvg_res_small[:,0], ax=ax)
plt.vlines(x=0.4, ymin=0, ymax=max(fo_tvg_res_small[:,1]*1000), color='purple', linestyles='dashed')
plt.xlabel('Gamma')
ax.text(0.5, 3, 'Threshold G=0.4', c='purple')
plt.ylabel('Convg. Time (ms)')
plt.savefig('VI Forest small time v gamma')

sns.set()
fig, ax = plt.subplots(figsize=(3, 2))
plt.subplots_adjust(bottom=.24)
plt.subplots_adjust(left=.2)
plt.title('16 Convg. Iter. vs Gamma')
sns.lineplot(y=fo_tvg_res_small[:,3], x=fo_tvg_res_small[:,0], ax=ax)
plt.vlines(x=0.4, ymin=0, ymax=max(fo_tvg_res_small[:,3]), color='purple', linestyles='dashed')
plt.xlabel('Gamma')
ax.text(0.5, 200, 'Threshold G=0.4', c='purple')
plt.ylabel('Convg. Iterations')
plt.savefig('VI Forest small iter v gamma')


sns.set()
fig, ax = plt.subplots(figsize=(3, 2))
plt.subplots_adjust(bottom=.24)
plt.subplots_adjust(left=.24)
plt.title('2500 Convg. Time vs Gamma')
sns.lineplot(y=fo_tvg_res_big[:,1]*1000, x=fo_tvg_res_big[:,0], ax=ax)
plt.vlines(x=0.995, ymin=0, ymax=max(fo_tvg_res_big[:,1]*1000), color='purple', linestyles='dashed')
ax.text(0.25, 3000, 'Threshold G=0.995', c='purple')
plt.xlabel('Gamma')
plt.ylabel('Convg. Time (ms)')
plt.savefig('VI Forest big time v gamma')


sns.set()
fig, ax = plt.subplots(figsize=(3, 2))
plt.subplots_adjust(bottom=.24)
plt.subplots_adjust(left=.24)
plt.title('2500 Convg. Time vs Gamma')
sns.lineplot(y=fo_tvg_res_big[:,1]*1000, x=1/(1-fo_tvg_res_big[:,0])*np.log(1/(1-fo_tvg_res_big[:,0])), ax=ax)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Gamma')
plt.ylabel('Convg. Time (ms)')
plt.savefig('VI Forest big time v gamma')

sns.set()
fig, ax = plt.subplots(figsize=(3, 2))
plt.subplots_adjust(bottom=.24)
plt.subplots_adjust(left=.24)
plt.title('2500 Convg. Iter. vs Gamma')
sns.lineplot(y=fo_tvg_res_big[:,3], x=fo_tvg_res_big[:,0], ax=ax)
plt.vlines(x=0.995, ymin=0, ymax=max(fo_tvg_res_big[:,3]), color='purple', linestyles='dashed')
ax.text(0.25, 2000, 'Threshold G=0.995', c='purple')
plt.xlabel('Gamma')
plt.ylabel('Convg. Iterations')
plt.savefig('VI Forest big iter v gamma')

sns.set()
fig, ax = plt.subplots(figsize=(3, 2.5))
ax2 = ax.twinx()
ax.grid(False)
ax2.grid(False)
ax.set_ylabel('Time (sec)', color='blue')
ax2.set_ylabel('Iterations', color='green')
ax.tick_params(axis='y', labelsize=9.5, color = 'blue', labelcolor='blue')
ax2.tick_params(axis='y', labelsize=9.5, color = 'green', labelcolor='green')
ax2.set_ylim([0,20000])
plt.subplots_adjust(bottom=.22)
plt.subplots_adjust(left=.22)
plt.subplots_adjust(right=.74)
plt.title('Forest Convg. vs State Size')
sns.lineplot(y=tvs_res[:,1], x=tvs_res[:,0], ax=ax, color='blue')
sns.lineplot(y=tvs_res[:,2], x=tvs_res[:,0], ax=ax2, color='green')
ax.set_xlabel('State Size')
ax.tick_params(axis='x', labelsize=9.5)
ax.text(0, 50, 'G=0.9995', c='purple')
ax.text(0, 25, 'P=0.0001', c='purple')
plt.savefig('VI Forest conv v states')

sns.set()
fig, ax = plt.subplots(figsize=(2.1, 2.1))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.24)
plt.title('Forest Time per Iteration')
ax.set_ylabel('Time per Iteration (ms)', color='blue')
sns.lineplot(y=tvs_res[:,1]/tvs_res[:,2]*1000, x=tvs_res[:,0], ax=ax, color='blue')
ax.set_xlabel('State Size')
ax.tick_params(axis='y', labelsize=10, color = 'blue', labelcolor='blue')
plt.savefig('VI Forest conv2 v states')