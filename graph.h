#ifndef OLC_PGE_APPLICATION
#define OLC_PGE_APPLICATION

#include <vector>
#include <thread>
#include "olcPixelGameEngine/olcPixelGameEngine.h"

#include <stdio.h>
#include <iostream>

using namespace std;



struct Point {
	double x, y;
	olc::Pixel color = olc::BLACK;

	Point(double x = 0, double y = 0) : x(x), y(y) {
		
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











struct Graph : public olc::PixelGameEngine {

private:

	double minX, maxX, minY, maxY;
	std::thread graphThread;

	bool dragging = false;
	Point dragStart;

public:

	int w, h, pixelW, pixelH;

	vector<Point> points;
	vector<Line*> lines;


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


		Point mousePos = pointToWorldSpace(Point(GetMouseX(), GetMouseY()));

		float zoom = std::exp(GetMouseWheel() * -0.001);

		// translate so mouse pos is at origin, than scale, than translate back
		// this ensures that the point the mouse hoovers doesn't move when scaling
		minX = mousePos.x + (minX - mousePos.x) * zoom;
		maxX = mousePos.x + (maxX - mousePos.x) * zoom;
		minY = mousePos.y + (minY - mousePos.y) * zoom;
		maxY = mousePos.y + (maxY - mousePos.y) * zoom;

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
			drawPoint(points[i], 2);
		}

		return true;
	}

	void addLine(Line* line) {
		lines.push_back(line);
	}

	void drawLine(Line* line) {
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

	void drawPoint(Point& p, int size = 1) {
		Point screenSpace = pointToScreenSpace(p);
		FillCircle(screenSpace.x, screenSpace.y, size, p.color);
	}


	void waitFinish() {
		graphThread.join();
	}


	void drawAxis() {
		int x0 = -minX * (w / (maxX - minX));
		DrawLine(x0, 0, x0, h, olc::BLACK);

		int y0 = h + minY * (h / (maxY - minY));
		DrawLine(0, y0, w, y0, olc::BLACK);
	}


	Point pointToScreenSpace(const Point& p) {
		// we need to map X coordinate from [minX, maxX] to [0, w] (and similar with Y)

		Point screenSpace;
		screenSpace.x = (p.x - minX) * (w / (maxX - minX));
		screenSpace.y = h - (p.y - minY) * (h / (maxY - minY));

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
