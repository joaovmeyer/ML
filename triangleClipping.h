#ifndef TRIANGLE_CLIPPING_H
#define TRIANGLE_CLIPPING_H

#include <vector>



struct Point {
	double x, y, r;
	olc::Pixel color;

	Point(double x = 0, double y = 0, double radius = 2, const olc::Pixel color = olc::BLACK) : x(x), y(y), r(radius), color(color) {
		
	}
};


struct Point3D {
	double x, y, z, r;
	olc::Pixel color;

	Point3D(double x = 0, double y = 0, double z = 0, double radius = 2, const olc::Pixel color = olc::BLACK) : x(x), y(y), z(z), r(radius), color(color) {
		
	}
};



double lerp(double a, double b, double t) {
	return a + (b - a) * t;
}

double invLerp(double a, double b, double c) {
	return (c - a) / (b - a);
}

Point lerp(const Point& a, const Point& b, double t) {
	return Point(
		a.x + (b.x - a.x) * t,
		a.y + (b.y - a.y) * t,
		a.r + (b.r - a.r) * t,
		PixelLerp(a.color, b.color, t)
	);
}

Point3D lerp(const Point3D& a, const Point3D& b, double t) {
	return Point3D(
		a.x + (b.x - a.x) * t,
		a.y + (b.y - a.y) * t,
		a.z + (b.z - a.z) * t,
		a.r + (b.r - a.r) * t,
		PixelLerp(a.color, b.color, t)
	);
}

std::vector<std::vector<Point>> clipToLeftWall(const std::vector<std::vector<Point>>& triangles) {
	std::vector<std::vector<Point>> newTriangles;

	for (size_t i = 0; i < triangles.size(); ++i) {
		std::vector<Point> outsidePoints;
		std::vector<Point> insidePoints;

		std::vector<Point> tri = triangles[i];

		if (tri[0].x <= 0.0) {
			outsidePoints.push_back(tri[0]);
		} else {
			insidePoints.push_back(tri[0]);
		}
		if (tri[1].x <= 0.0) {
			outsidePoints.push_back(tri[1]);
		} else {
			insidePoints.push_back(tri[1]);
		}
		if (tri[2].x <= 0.0) {
			outsidePoints.push_back(tri[2]);
		} else {
			insidePoints.push_back(tri[2]);
		}

		if (insidePoints.size() == 3) {
			newTriangles.push_back(tri);
		} else if (insidePoints.size() == 1) {
			double t1 = invLerp(insidePoints[0].x, outsidePoints[0].x, 0.0);
			double t2 = invLerp(insidePoints[0].x, outsidePoints[1].x, 0.0);

			newTriangles.push_back({
				insidePoints[0],
				lerp(insidePoints[0], outsidePoints[0], t1),
				lerp(insidePoints[0], outsidePoints[1], t2)
			});
		} else if (insidePoints.size() == 2) {
			double t1 = invLerp(outsidePoints[0].x, insidePoints[0].x, 0.0);
			double t2 = invLerp(outsidePoints[0].x, insidePoints[1].x, 0.0);

			newTriangles.push_back({
				insidePoints[0],
				insidePoints[1],
				lerp(outsidePoints[0], insidePoints[0], t1)
			});
			newTriangles.push_back({
				insidePoints[1],
				newTriangles.back()[2],
				lerp(outsidePoints[0], insidePoints[1], t2)
			});
		}
	}

	return newTriangles;
}


std::vector<std::vector<Point>> clipToRightWall(const std::vector<std::vector<Point>>& triangles, double w) {
	std::vector<std::vector<Point>> newTriangles;

	for (size_t i = 0; i < triangles.size(); ++i) {
		std::vector<Point> outsidePoints;
		std::vector<Point> insidePoints;

		std::vector<Point> tri = triangles[i];

		if (tri[0].x >= w - 1.0) {
			outsidePoints.push_back(tri[0]);
		} else {
			insidePoints.push_back(tri[0]);
		}
		if (tri[1].x >= w - 1.0) {
			outsidePoints.push_back(tri[1]);
		} else {
			insidePoints.push_back(tri[1]);
		}
		if (tri[2].x >= w - 1.0) {
			outsidePoints.push_back(tri[2]);
		} else {
			insidePoints.push_back(tri[2]);
		}

		if (insidePoints.size() == 3) {
			newTriangles.push_back(tri);
		} else if (insidePoints.size() == 1) {
			double t1 = invLerp(insidePoints[0].x, outsidePoints[0].x, w - 1.0);
			double t2 = invLerp(insidePoints[0].x, outsidePoints[1].x, w - 1.0);

			newTriangles.push_back({
				insidePoints[0],
				lerp(insidePoints[0], outsidePoints[0], t1),
				lerp(insidePoints[0], outsidePoints[1], t2)
			});
		} else if (insidePoints.size() == 2) {
			double t1 = invLerp(outsidePoints[0].x, insidePoints[0].x, w - 1.0);
			double t2 = invLerp(outsidePoints[0].x, insidePoints[1].x, w - 1.0);

			newTriangles.push_back({
				insidePoints[0],
				insidePoints[1],
				lerp(outsidePoints[0], insidePoints[0], t1)
			});
			newTriangles.push_back({
				insidePoints[1],
				newTriangles.back()[2],
				lerp(outsidePoints[0], insidePoints[1], t2)
			});
		}
	}

	return newTriangles;
}


std::vector<std::vector<Point>> clipToTopWall(const std::vector<std::vector<Point>>& triangles) {
	std::vector<std::vector<Point>> newTriangles;

	for (size_t i = 0; i < triangles.size(); ++i) {
		std::vector<Point> outsidePoints;
		std::vector<Point> insidePoints;

		std::vector<Point> tri = triangles[i];

		if (tri[0].y < 0.0) {
			outsidePoints.push_back(tri[0]);
		} else {
			insidePoints.push_back(tri[0]);
		}
		if (tri[1].y < 0.0) {
			outsidePoints.push_back(tri[1]);
		} else {
			insidePoints.push_back(tri[1]);
		}
		if (tri[2].y < 0.0) {
			outsidePoints.push_back(tri[2]);
		} else {
			insidePoints.push_back(tri[2]);
		}

		if (insidePoints.size() == 3) {
			newTriangles.push_back(tri);
		} else if (insidePoints.size() == 1) {
			double t1 = invLerp(insidePoints[0].y, outsidePoints[0].y, 0.0);
			double t2 = invLerp(insidePoints[0].y, outsidePoints[1].y, 0.0);

			newTriangles.push_back({
				insidePoints[0],
				lerp(insidePoints[0], outsidePoints[0], t1),
				lerp(insidePoints[0], outsidePoints[1], t2)
			});
		} else if (insidePoints.size() == 2) {
			double t1 = invLerp(outsidePoints[0].y, insidePoints[0].y, 0.0);
			double t2 = invLerp(outsidePoints[0].y, insidePoints[1].y, 0.0);

			newTriangles.push_back({
				insidePoints[0],
				insidePoints[1],
				lerp(outsidePoints[0], insidePoints[0], t1)
			});
			newTriangles.push_back({
				insidePoints[1],
				newTriangles.back()[2],
				lerp(outsidePoints[0], insidePoints[1], t2)
			});
		}
	}

	return newTriangles;
}

std::vector<std::vector<Point>> clipToBottomWall(const std::vector<std::vector<Point>>& triangles, double h) {
	std::vector<std::vector<Point>> newTriangles;

	for (size_t i = 0; i < triangles.size(); ++i) {
		std::vector<Point> outsidePoints;
		std::vector<Point> insidePoints;

		std::vector<Point> tri = triangles[i];

		if (tri[0].y >= h) {
			outsidePoints.push_back(tri[0]);
		} else {
			insidePoints.push_back(tri[0]);
		}
		if (tri[1].y >= h) {
			outsidePoints.push_back(tri[1]);
		} else {
			insidePoints.push_back(tri[1]);
		}
		if (tri[2].y >= h) {
			outsidePoints.push_back(tri[2]);
		} else {
			insidePoints.push_back(tri[2]);
		}

		if (insidePoints.size() == 3) {
			newTriangles.push_back(tri);
		} else if (insidePoints.size() == 1) {
			double t1 = invLerp(insidePoints[0].y, outsidePoints[0].y, h - 1.0);
			double t2 = invLerp(insidePoints[0].y, outsidePoints[1].y, h - 1.0);

			newTriangles.push_back({
				insidePoints[0],
				lerp(insidePoints[0], outsidePoints[0], t1),
				lerp(insidePoints[0], outsidePoints[1], t2)
			});
		} else if (insidePoints.size() == 2) {
			double t1 = invLerp(outsidePoints[0].y, insidePoints[0].y, h - 1.0);
			double t2 = invLerp(outsidePoints[0].y, insidePoints[1].y, h - 1.0);

			newTriangles.push_back({
				insidePoints[0],
				insidePoints[1],
				lerp(outsidePoints[0], insidePoints[0], t1)
			});
			newTriangles.push_back({
				insidePoints[1],
				newTriangles.back()[2],
				lerp(outsidePoints[0], insidePoints[1], t2)
			});
		}
	}

	return newTriangles;
}










std::vector<std::vector<Point>> clipTriangle(const std::vector<Point>& triangle, double w, double h) {
	std::vector<std::vector<Point>> newTriangles = { triangle };

	newTriangles = clipToLeftWall(newTriangles);
	newTriangles = clipToRightWall(newTriangles, w);
	newTriangles = clipToTopWall(newTriangles);
	newTriangles = clipToBottomWall(newTriangles, h);

	return newTriangles;
}












std::vector<std::vector<Point3D>> clipToNearPlane(const std::vector<std::vector<Point3D>>& triangles, double nearPlane = 0.01) {
	std::vector<std::vector<Point3D>> newTriangles;

	for (size_t i = 0; i < triangles.size(); ++i) {
		std::vector<Point3D> outsidePoints;
		std::vector<Point3D> insidePoints;

		std::vector<Point3D> tri = triangles[i];

		if (tri[0].z <= nearPlane) {
			outsidePoints.push_back(tri[0]);
		} else {
			insidePoints.push_back(tri[0]);
		}
		if (tri[1].z <= nearPlane) {
			outsidePoints.push_back(tri[1]);
		} else {
			insidePoints.push_back(tri[1]);
		}
		if (tri[2].z <= nearPlane) {
			outsidePoints.push_back(tri[2]);
		} else {
			insidePoints.push_back(tri[2]);
		}

		if (insidePoints.size() == 3) {
			newTriangles.push_back(tri);
		} else if (insidePoints.size() == 1) {
			double t1 = invLerp(insidePoints[0].z, outsidePoints[0].z, nearPlane);
			double t2 = invLerp(insidePoints[0].z, outsidePoints[1].z, nearPlane);

			newTriangles.push_back({
				insidePoints[0],
				lerp(insidePoints[0], outsidePoints[0], t1),
				lerp(insidePoints[0], outsidePoints[1], t2)
			});
		} else if (insidePoints.size() == 2) {
			double t1 = invLerp(outsidePoints[0].z, insidePoints[0].z, nearPlane);
			double t2 = invLerp(outsidePoints[0].z, insidePoints[1].z, nearPlane);

			newTriangles.push_back({
				insidePoints[0],
				insidePoints[1],
				lerp(outsidePoints[0], insidePoints[0], t1)
			});
			newTriangles.push_back({
				insidePoints[1],
				newTriangles.back()[2],
				lerp(outsidePoints[0], insidePoints[1], t2)
			});
		}
	}

	return newTriangles;
}





#endif
