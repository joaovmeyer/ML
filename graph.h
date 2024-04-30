#ifndef OLC_PGE_APPLICATION
#define OLC_PGE_APPLICATION

#include <vector>
#include <thread>
#include "olcPixelGameEngine/olcPixelGameEngine.h"

#include <stdio.h>
#include <iostream>

using namespace std;



struct Point {
	double x, y, r;
	olc::Pixel color;

	Point(double x = 0, double y = 0, double radius = 2, const olc::Pixel color = olc::BLACK) : x(x), y(y), r(radius), color(color) {
		
	}
};

struct Line {
	vector<Point> points;
	olc::Pixel color;

	Line(olc::Pixel color = olc::BLACK) : color(color) {

	}

	void addPoint(const Point& p) {
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
		double slopeX = (img.sprite->width - 1) / (p2[0] - p1[0]);
		if (startX > endX) std::swap(startX, endX);

		int startY = std::min(std::max(static_cast<int>(p1[1]), 0), h);
		int endY = std::min(std::max(static_cast<int>(p2[1]), 0), h);
		double slopeY = (img.sprite->height - 1) / (p1[1] - p2[1]);
		if (startY > endY) std::swap(startY, endY);

		for (int y = startY; y < endY; ++y) {
			int imgY = (y - p2[1]) * slopeY;

			for (int x = startX; x < endX; ++x) {
				int imgX = (x - p1[0]) * slopeX;

				Draw(x, y, img.sprite->GetPixel(imgX, imgY));
			}
		}
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







#endif
