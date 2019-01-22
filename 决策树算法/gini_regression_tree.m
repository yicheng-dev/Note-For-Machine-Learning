load carsmall
X = [Horsepower Weight];
rtree = fitrtree(X, MPG);
view (rtree)