import numpy as np
from scipy.spatial import Delaunay, ConvexHull


def distance(ptA, ptB):
	return np.linalg.norm(np.subtract(ptA, ptB))

def order_walls(walls):
	new_wall = walls.pop(0)
	while walls:
		add = [wall for wall in walls if new_wall[-1] in wall][0]
		walls.remove(add)
		add.remove(new_wall[-1])
		new_wall.extend(add)
	return new_wall

def getTriangleArea(A, B, C):
	return .5 * abs(A[0]*(B[1] - C[1]) + B[0]*(C[1] - A[1]) + C[0]*(A[1] - B[1]))

def concaveArea(points):
	tri = Delaunay(points)
	outerwalls = tri.convex_hull.tolist()
	outerwalls = order_walls(outerwalls)
	verts = tri.vertices.tolist()
	change = False
	i = 0
	while i < len(outerwalls) - 1:
		at = outerwalls[i]
		next = outerwalls[i + 1]
		outer_dist = distance(points[at], points[next])
		inner = None
		for t in verts:
			inners = set(t) ^ {at, next}
			if len(inners) == 1 and len(set(outerwalls) & set(t)) == 2:
				inner = inners.pop()
				break
		if inner != None and outer_dist > distance(points[at], points[inner]):
			outerwalls.insert(i+1, inner)
			change = True
			verts.remove(t)
			i += 1
		i += 1
		if i >= len(outerwalls) - 1 and change:
			change = False
			i = 0
	pts = np.array([points[i] for i in outerwalls])
	return sum(map(lambda vs: getTriangleArea(*[points[i] for i in vs]), verts))


def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6

def convex_volume(points):
    ch = ConvexHull(points)
    simplices = np.column_stack((np.repeat(ch.vertices[0], ch.nsimplex), ch.simplices))
    tets = ch.points[simplices]
    return np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1],
                                     tets[:, 2], tets[:, 3]))
