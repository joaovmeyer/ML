#ifndef OLC_PGE_APPLICATION
#define OLC_PGE_APPLICATION

#include <vector>
#include <thread>
#include "olcPixelGameEngine/olcPixelGameEngine.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <cmath>
#include <functional>

#include "triangleClipping.h"

using namespace std;





struct Line {
	vector<Point> points;
	olc::Pixel color;

	Line(olc::Pixel color = olc::BLACK) : color(color) {

	}

	void addPoint(const Point& p) {
		points.push_back(p);
	}
};

struct Line3D {
	std::vector<Point3D> points;
	olc::Pixel color;

	Line3D(olc::Pixel color = olc::BLACK) : color(color) {

	}

	void addPoint(const Point3D& p) {
		points.push_back(p);
	}
};

struct Image {
	olc::Sprite* sprite;
	double x0, y0, x1, y1;

	Image(olc::Sprite* s, double minX, double minY, double maxX, double maxY) {
		sprite = s;
		x0 = minX; y0 = minY; x1 = maxX; y1 = maxY;
	}
};

struct Text {
	string str;
	double x = 0, y = 0;

	Text(const string& str, double x, double y) : str(str), x(x), y(y) {

	}
};









struct Graph : public olc::PixelGameEngine {

private:

	double minX, maxX, minY, maxY;
	std::thread graphThread;

	bool dragging = false;
	Point dragStart;

public:

	int w, h, pixelW, pixelH;

	vector<Point> points;
	vector<std::shared_ptr<Line>> lines;
	vector<Image> images;
	vector<Text> texts;


	Graph(int w = 800, int h = 650, int pixelW = 1, int pixelH = 1) : w(w), h(h), pixelW(pixelW), pixelH(pixelH) {

		minX = -3;
		maxX = 7;
		minY = -3 * ((double) h / w);
		maxY = 7 * ((double) h / w);

		// start the graph in another thread so the rest of the code doesn't stop
		graphThread = std::thread([=]() {
			if (Construct(w, h, pixelW, pixelH)) {
				Start();
			}
		});
	}

	bool OnUserCreate() override {
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override {
		
		Clear(olc::WHITE);

		// enable alpha blending
		SetPixelMode(olc::Pixel::ALPHA);

		Point mousePos = pointToWorldSpace(Point(GetMouseX(), GetMouseY()));

		bool zoomX = GetKey(olc::Key::X).bHeld, zoomY = GetKey(olc::Key::Y).bHeld;
		float zoom = std::exp(GetMouseWheel() * -0.001);

		// translate so mouse pos is at origin, than scale, than translate back
		// this ensures that the point the mouse hoovers doesn't move when scaling
		if (zoomX || !zoomY) {
			minX = mousePos.x + (minX - mousePos.x) * zoom;
			maxX = mousePos.x + (maxX - mousePos.x) * zoom;
		}
		if (zoomY || !zoomX) {
			minY = mousePos.y + (minY - mousePos.y) * zoom;
			maxY = mousePos.y + (maxY - mousePos.y) * zoom;
		}

		// move view
		if (GetMouse(0).bPressed && !dragging) {
			dragStart = mousePos;
			dragging = true;
		} else if (GetMouse(0).bReleased) {
			dragging = false;
		}

		if (dragging) {
			minX -= (mousePos.x - dragStart.x);
			maxX -= (mousePos.x - dragStart.x);
			minY -= (mousePos.y - dragStart.y);
			maxY -= (mousePos.y - dragStart.y);

			dragStart = pointToWorldSpace(Point(GetMouseX(), GetMouseY()));
		}


		drawAxis();

		for (size_t i = 0; i < lines.size(); ++i) {
			drawLine(lines[i]);
		}

		for (size_t i = 0; i < points.size(); ++i) {
			drawPoint(points[i]);
		}

		for (size_t i = 0; i < images.size(); ++i) {
			drawImage(images[i]);
		}

		for (size_t i = 0; i < texts.size(); ++i) {
			drawText(texts[i]);
		}

		return true;
	}

	void addLine(const std::shared_ptr<Line>& line) {
		lines.push_back(line);
	}
	void addLine(const Line& line) {
		lines.push_back(std::make_shared<Line>(line));
	}

	void drawLine(const std::shared_ptr<Line>& line) {
		Line l = *line;
		for (size_t i = 0; i + 1 < l.points.size(); ++i) {
			// transform to screen space
			Point screenSpaceP1 = pointToScreenSpace(l.points[i]);
			Point screenSpaceP2 = pointToScreenSpace(l.points[i + 1]);

			olc::vi2d p1(screenSpaceP1.x, screenSpaceP1.y);
			olc::vi2d p2(screenSpaceP2.x, screenSpaceP2.y);

			if (ClipLineToScreen(p1, p2)) {
				DrawLine(p1, p2, l.color);
			}
		}
	}

	void addPoint(const Point& p) {
		points.push_back(p);
	}

	void drawPoint(Point& p) {
		Point screenSpace = pointToScreenSpace(p);
		FillCircle(screenSpace.x, screenSpace.y, p.r, p.color);
	}


	void addImage(olc::Sprite* sprite, double minX = 0, double minY = 0, double maxX = 1, double maxY = 1) {
		images.push_back(Image(sprite, minX, minY, maxX, maxY));
	}

	void drawImage(const Image& img) {
		vector<double> p1 = pointToScreenSpace(img.x0, img.y0);
		vector<double> p2 = pointToScreenSpace(img.x1, img.y1);

		int startX = std::min(std::max(static_cast<int>(p1[0]), 0), w);
		int endX = std::min(std::max(static_cast<int>(p2[0]), 0), w);
		double slopeX = (img.sprite->width) / (p2[0] - p1[0]);
		if (startX > endX) std::swap(startX, endX);

		int startY = std::min(std::max(static_cast<int>(p1[1]), 0), h);
		int endY = std::min(std::max(static_cast<int>(p2[1]), 0), h);
		double slopeY = (img.sprite->height) / (p1[1] - p2[1]);
		if (startY > endY) std::swap(startY, endY);

		for (int y = startY; y < endY; ++y) {
			int imgY = (y - p2[1]) * slopeY;

			for (int x = startX; x < endX; ++x) {
				int imgX = (x - p1[0]) * slopeX;

				Draw(x, y, img.sprite->GetPixel(imgX, imgY));
			}
		}
	}


	void addText(const string& text, double x = 0, double y = 0) {
		texts.push_back(Text(text, x, y));
	}

	void drawText(const Text& text) {
		vector<double> p = pointToScreenSpace(text.x, text.y);
		DrawString(p[0], p[1], text.str, olc::BLACK, 1);
	}


	

	void drawAxis() {
		int x0 = -minX * (w / (maxX - minX));
		DrawLine(x0, 0, x0, h, olc::BLACK);

		int y0 = h + minY * (h / (maxY - minY));
		DrawLine(0, y0, w, y0, olc::BLACK);
	}


	void waitFinish() {
		graphThread.join();
	}


	Point pointToScreenSpace(const Point& p) {
		// we need to map X coordinate from [minX, maxX] to [0, w] (and similar with Y)

		Point screenSpace;
		screenSpace.x = (p.x - minX) * (w / (maxX - minX));
		screenSpace.y = h - (p.y - minY) * (h / (maxY - minY));

		return screenSpace;
	}

	vector<double> pointToScreenSpace(double x, double y) {
		// we need to map X coordinate from [minX, maxX] to [0, w] (and similar with Y)

		vector<double> screenSpace(2);
		screenSpace[0] = (x - minX) * (w / (maxX - minX));
		screenSpace[1] = h - (y - minY) * (h / (maxY - minY));

		return screenSpace;
	}

	Point pointToWorldSpace(const Point& p) {
		// we need to map X coordinate from [0, w] to [minX, maxX] (and similar with Y)

		Point worldSpace;
		worldSpace.x = p.x / (w / (maxX - minX)) + minX;
		worldSpace.y = (maxY - minY) - p.y / (h / (maxY - minY)) + minY;

		return worldSpace;
	}

};
















struct Graph3D : public olc::PixelGameEngine {

private:

	double minX, maxX, minY, maxY;
	std::thread graphThread;

	bool dragging = false;
	Point dragStart;

	double yaw, pitch;
	double nearPlane;
	double cameraDst;

public:

	int w, h, pixelW, pixelH;

	vector<Point3D> points;
	vector<Line3D> lines;


	Graph3D(int w = 800, int h = 650, int pixelW = 1, int pixelH = 1) : w(w), h(h), pixelW(pixelW), pixelH(pixelH) {

		minX = -7;
		maxX = 7;
		minY = -7 * ((double) h / w);
		maxY = 7 * ((double) h / w);

		yaw = 0.0;
		pitch = 0.0;
		nearPlane = 0.001;

		cameraDst = -2.0;

		// start the graph in another thread so the rest of the code doesn't stop
		graphThread = std::thread([=]() {
			if (Construct(w, h, pixelW, pixelH)) {
				Start();
			}
		});
	}

	bool OnUserCreate() override {
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override {
		
		Clear(olc::WHITE);

		// enable alpha blending
		SetPixelMode(olc::Pixel::ALPHA);

		Point mousePos = pointToWorldSpace(Point3D(GetMouseX(), GetMouseY(), 1));

		// move view
		if (GetMouse(0).bPressed && !dragging) {
			dragStart = mousePos;
			dragging = true;
		} else if (GetMouse(0).bReleased) {
			dragging = false;
		}


		// move camera
		cameraDst = std::min(cameraDst - std::exp(GetMouseWheel() * -0.001) + 1.0, 0.0);


		if (dragging) {

			yaw -= (mousePos.x - dragStart.x);
			pitch += (mousePos.y - dragStart.y);

			dragStart = pointToWorldSpace(Point3D(GetMouseX(), GetMouseY(), 1));
		}


		drawAxis();

		for (size_t i = 0; i < lines.size(); ++i) {
			drawLine(lines[i]);
		}

		for (size_t i = 0; i < points.size(); ++i) {
			drawPoint(points[i]);
		}

		teste();

		return true;
	}


	// plotting a function. Still need to figure out depthbuffer and a basic occlusion culling
	void teste() {

		std::function<double(double, double)> f = [](double x, double z) {
			return std::sin(x) + std::sin(z);
		};

		vector<vector<Point3D>> heightMap;

		double step = 0.5;

		double maxValue = -99999;
		double minValue = 99999;

		for (double x = -5.0; x < 5.0; x += step) {

			vector<Point3D> v;

			for (double z = -5.0; z < 5.0; z += step) {
				double y = f(x, z);

				maxValue = std::max(maxValue, y);
				minValue = std::min(minValue, y);

				v.push_back(Point3D(x, y, z));
			}

			heightMap.push_back(v);
		}

		olc::Pixel colorHigh = olc::RED;
		olc::Pixel colorLow = olc::BLUE;

		for (size_t i = 0; i < heightMap.size() - 1; ++i) {
			for (size_t j = 0; j < heightMap[0].size() - 1; ++j) {

				Point3D p1 = pointToCameraSpace(heightMap[i][j]);
				Point3D p2 = pointToCameraSpace(heightMap[i + 1][j]);
				Point3D p3 = pointToCameraSpace(heightMap[i + 1][j + 1]);
				Point3D p4 = pointToCameraSpace(heightMap[i][j + 1]);

				p1.color = PixelLerp(colorLow, colorHigh, (heightMap[i][j].y - minValue) / (maxValue - minValue));
				p2.color = PixelLerp(colorLow, colorHigh, (heightMap[i + 1][j].y - minValue) / (maxValue - minValue));
				p3.color = PixelLerp(colorLow, colorHigh, (heightMap[i + 1][j + 1].y - minValue) / (maxValue - minValue));
				p4.color = PixelLerp(colorLow, colorHigh, (heightMap[i][j + 1].y - minValue) / (maxValue - minValue));

				olc::vf2d z(0, 0);

			//	vector<vector<Point3D>> tris = clipToNearPlane({ { p1, p2, p3 }, { p1, p4, p3 } }, nearPlane);
				vector<vector<Point3D>> tris = { { p1, p2, p3 }, { p1, p4, p3 } };

				for (size_t i = 0; i < tris.size(); ++i) {
					Point screenSpace1 = pointToScreenSpace(tris[i][0]);
					Point screenSpace2 = pointToScreenSpace(tris[i][1]);
					Point screenSpace3 = pointToScreenSpace(tris[i][2]);

					vector<vector<Point>> newTris = clipTriangle({ screenSpace1, screenSpace2, screenSpace3 }, w, h);

					for (size_t j = 0; j < newTris.size(); ++j) {

						olc::vf2d newTriP1(newTris[j][0].x, newTris[j][0].y);
						olc::vf2d newTriP2(newTris[j][1].x, newTris[j][1].y);
						olc::vf2d newTriP3(newTris[j][2].x, newTris[j][2].y);

						FillTexturedTriangle({ newTriP1, newTriP2, newTriP3 }, { z, z, z }, { newTris[j][0].color, newTris[j][1].color, newTris[j][2].color }, nullptr);
					}
				}
			}
		}

	}


	void addLine(const Line3D& line) {
		lines.push_back(line);
	}

	void drawLine(const Line3D& line, bool expand = false) {
		Line3D l = line;
		for (size_t i = 0; i + 1 < l.points.size(); ++i) {

			// transform to camera space
			Point3D cameraSpaceP1 = pointToCameraSpace(l.points[i]);
			Point3D cameraSpaceP2 = pointToCameraSpace(l.points[i + 1]);

			// near plane clipping
			if (cameraSpaceP1.z <= nearPlane) {

				// both points behind near plane
				if (cameraSpaceP2.z <= nearPlane) continue;

				double t = (cameraSpaceP2.z - nearPlane) / (cameraSpaceP2.z - cameraSpaceP1.z);

				cameraSpaceP1.x = cameraSpaceP2.x + (cameraSpaceP1.x - cameraSpaceP2.x) * t;
				cameraSpaceP1.y = cameraSpaceP2.y + (cameraSpaceP1.y - cameraSpaceP2.y) * t;
				cameraSpaceP1.z = cameraSpaceP2.z + (cameraSpaceP1.z - cameraSpaceP2.z) * t;
			} else if (cameraSpaceP2.z <= nearPlane) {

				double t = (cameraSpaceP1.z - nearPlane) / (cameraSpaceP1.z - cameraSpaceP2.z);

				cameraSpaceP2.x = cameraSpaceP1.x + (cameraSpaceP2.x - cameraSpaceP1.x) * t;
				cameraSpaceP2.y = cameraSpaceP1.y + (cameraSpaceP2.y - cameraSpaceP1.y) * t;
				cameraSpaceP2.z = cameraSpaceP1.z + (cameraSpaceP2.z - cameraSpaceP1.z) * t;
			}

			// transform to screen space
			Point screenSpaceP1 = pointToScreenSpace(cameraSpaceP1);
			Point screenSpaceP2 = pointToScreenSpace(cameraSpaceP2);

			// expand will make the line segmend go through the second point untill the border (only here for drawAxis)
			if (expand) {
				double wH = w * 0.5, hH = h * 0.5;
				screenSpaceP2.x -= wH; screenSpaceP2.y -= hH;

				double mult = std::max(
					std::abs(screenSpaceP2.x) > 0.1 ? std::abs(wH / screenSpaceP2.x) : 0.0,
					std::abs(screenSpaceP2.y) > 0.1 ? std::abs(hH / screenSpaceP2.y) : 0.0
				);

				screenSpaceP2.x = screenSpaceP2.x * mult + wH; screenSpaceP2.y = screenSpaceP2.y * mult + hH;
			}


			DrawLine(screenSpaceP1.x, screenSpaceP1.y, screenSpaceP2.x, screenSpaceP2.y, l.color);

		}
	}

	void addPoint(const Point3D& p) {
		points.push_back(p);
	}

	void drawPoint(const Point3D& p) {
		Point3D cameraSpace = pointToCameraSpace(p);

		// behind near plane
		if (cameraSpace.z <= nearPlane) return;

		Point screenSpace = pointToScreenSpace(cameraSpace);
		FillCircle(screenSpace.x, screenSpace.y, screenSpace.r, p.color);
	}


	void drawAxis() {

		if (!cameraDst) return;

		Point3D origin(0.0, 0.0, 0.0);

		Line3D axisX1(olc::DARK_GREY); axisX1.addPoint(origin); axisX1.addPoint(Point3D(Point3D(-1, 0, 0)));
		Line3D axisX2(olc::DARK_GREY); axisX2.addPoint(origin); axisX2.addPoint(Point3D(Point3D(1, 0, 0)));

		Line3D axisY1(olc::DARK_GREY); axisY1.addPoint(origin); axisY1.addPoint(Point3D(Point3D(0, -1, 0)));
		Line3D axisY2(olc::DARK_GREY); axisY2.addPoint(origin); axisY2.addPoint(Point3D(Point3D(0, 1, 0)));

		Line3D axisZ1(olc::DARK_GREY); axisZ1.addPoint(origin); axisZ1.addPoint(Point3D(Point3D(0, 0, -1)));
		Line3D axisZ2(olc::DARK_GREY); axisZ2.addPoint(origin); axisZ2.addPoint(Point3D(Point3D(0, 0, 1)));

		drawLine(axisX1, true); drawLine(axisX2, true);
		drawLine(axisY1, true); drawLine(axisY2, true);
		drawLine(axisZ1, true); drawLine(axisZ2, true);
	}


	void waitFinish() {
		graphThread.join();
	}


	Point pointToScreenSpace(const Point3D& p) {
		// we need to map X coordinate from [minX, maxX] to [0, w] (and similar with Y)

		double FOV = 3.141592653 / 4.0;

	//	double perspectiveDiv = 1.0 / (p.z * std::tan(FOV * 0.5));
		double perspectiveDiv = -1.0 / cameraDst; // orthogonal perspective

		Point screenSpace;
		screenSpace.x = (p.x * perspectiveDiv - minX) * (w / (maxX - minX));
		screenSpace.y = h - (p.y * perspectiveDiv - minY) * (h / (maxY - minY));
		screenSpace.r = p.r;
		screenSpace.color = p.color;

		return screenSpace;
	}

	vector<double> pointToScreenSpace(double x, double y) {
		// we need to map X coordinate from [minX, maxX] to [0, w] (and similar with Y)

		vector<double> screenSpace(2);
		screenSpace[0] = (x - minX) * (w / (maxX - minX));
		screenSpace[1] = h - (y - minY) * (h / (maxY - minY));

		return screenSpace;
	}

	Point pointToWorldSpace(const Point3D& p) {
		// we need to map X coordinate from [0, w] to [minX, maxX] (and similar with Y)

		Point worldSpace;
		worldSpace.x = p.x / (w / (maxX - minX)) + minX;
		worldSpace.y = (maxY - minY) - p.y / (h / (maxY - minY)) + minY;

		return worldSpace;
	}


	Point3D pointToCameraSpace(Point3D p) {
		double cos1 = std::cos(pitch);
		double sin1 = std::sin(pitch);

		double cos2 = std::cos(yaw);
		double sin2 = std::sin(yaw);


		// rotate along Y axis
		double rot1X = p.z * sin2 + p.x * cos2;
		double rot1Z = p.z * cos2 - p.x * sin2;

		// rotate alon X axis
		double rot2Y = p.y * cos1 - rot1Z * sin1;
		double rot2Z = p.y * sin1 + rot1Z * cos1;

		// update it's coordinates after doing all the rotations (and perform translation)
		p.x = rot1X; p.y = rot2Y; p.z = rot2Z - cameraDst;

		return p;
	}

};














#endif
