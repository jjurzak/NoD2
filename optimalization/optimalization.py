#f(x,y) = min(x,y) + max(x,y)

import numpy as np 
import matplotlib.pyplot as plt 

# x < y -> min=x max=y -> f(x,y) = x^2+y | nablaF = 2x,1
# x > y -> min=y max=y -> f(x,y) = y^2+x | nablaF = 1,2y 
# x = y, avg x^2 + x

def f(xy):
    x, y = xy[0], xy[1]
    if x < y:
        return x**2+y
    elif x > y:
        return y**2+x
    else:
        return x**2+x

    
def grad_f(xy):
    x, y = xy[0], xy[1]
    if x < y:
        return np.array([2*x, 1.0])
    elif x > y:
        return np.array([1, 2*y])
    else:
        return np.array([ (2*x + 1.0)/2.0, (1.0 + 2*y)/2.0 ])


    #opitmize

def optimize(optimizer, lr, init=np.array([2.0, -1]), steps=200):
    w = init.astype(float).copy()
    history = [w.copy()]
    v = np.zeros(2)
    s = np.zeros(2)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for t in range(1, steps+1):
        g = grad_f(w)
        if optimizer == 'gd':
            w = w - lr * g 
        elif optimizer == 'momentum':
            v = beta1 * v + lr * g 
            w = w - v 
        elif optimizer == 'rmsprop':
            rho = 0.9 
            s = rho * s + (1 - rho) * (g**2)
            w = w - lr / (np.sqrt(s) + eps) * g
        elif optimizer == 'adam':
            g = np.clip(g, -5.0, 5.0)
            v = beta1 * v + (1 - beta1) * g 
            s = beta2 * s + (1 - beta2) * g
            v_corr = v / (1 - beta1**t)
            s_corr = s / (1 - beta2**t)
            s_corr = np.maximum(s_corr, 1e-12)
            w = w - lr * v_corr / (np.sqrt(s_corr) + eps)
            w = np.clip(w, -5.0, 5.0)

            if np.isnan(w).any():
                print(f'Adam zdetonowal pas szahida {t} (lr={lr})')
                history.append(w.copy())
                return np.array(history)
        else:
            raise ValueError("Unknown optimizer")
        history.append(w.copy())
    return np.array(history)


optimiers = ['gd', 'momentum', 'rmsprop', 'adam']
lrs = [0.025, 0.0025, 0.00025]
init = np.array([1.5, -0.5])
steps = 300

results = {}

for opt in optimiers:
    for lr in lrs:
        key = f'{opt}_lr{lr}'
        hist = optimize(opt, lr, init=init, steps=steps)
        losses = [f(p) for p in hist]
        results[key] = {'path' : hist, 'loss' : np.array(losses)}

xs = np.linspace(-2, 2, 400)
ys = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(xs, ys)
Z = np.vectorize(lambda a,b: f(np.array([a,b])))(X, Y)


plt.figure(figsize=(12,10))
CS = plt.contour(X,Y,Z, levels=30, cmap='gray')
plt.clabel(CS, inline=1, fontsize=8)
for key, val in results.items():
    path = val['path']
    plt.plot(path[:,0], path[:,1], label=key)
plt.scatter([init[0]], [init[1]], c='red', marker='x', label='start')
plt.legend(fontsize=8)
plt.title("Trajektorie optymalizacji dla roznych algorytmow i lr")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

plt.figure(figsize=(10,5))
for lr in lrs:
    key = f'adam_lr{lr}'
    plt.plot(results[key]['loss'], label=key)

plt.yscale('log')
plt.xlabel("iteracja")
plt.ylabel("f(x,y) (log scale)")
plt.legend()
plt.title("Porównanie strat (Adam) dla różnych lr")
plt.grid()
plt.show()

