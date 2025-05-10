/**
 * 人脸向量可视化图谱
 * 使用D3.js实现力导向图，展示人脸向量的相似关系
 */

// 全局变量
let graphData = null;
let simulation = null;
let svg = null;
let container = null;
let zoom = null;
let width = 0;
let height = 0;
let similarityThreshold = 0.7;

// 初始化函数
function init() {
  // 获取容器尺寸
  const graphContainer = document.getElementById("graph-container");
  width = graphContainer.clientWidth;
  height = graphContainer.clientHeight;

  // 创建SVG元素
  svg = d3.select("#graph-container").append("svg").attr("width", width).attr("height", height);

  // 创建缩放行为
  zoom = d3
    .zoom()
    .scaleExtent([0.1, 5])
    .on("zoom", (event) => {
      container.attr("transform", event.transform);
    });

  // 应用缩放行为到SVG
  svg.call(zoom);

  // 添加容器组，所有图形元素将添加到这个组
  container = svg.append("g");

  // 添加定义，用于存放图案
  const defs = svg.append("defs");

  // 初始化控件事件
  initControls();

  // 获取并展示数据
  fetchData();
}

// 初始化控件事件
function initControls() {
  // 相似度阈值滑块
  const thresholdSlider = document.getElementById("similarity-threshold");
  const thresholdValue = document.getElementById("threshold-value");

  // 设置滑块的最小值、最大值和步长
  thresholdSlider.min = "0.0";
  thresholdSlider.max = "1.0";
  thresholdSlider.step = "0.05";

  // 设置初始值
  similarityThreshold = 0.1;
  thresholdSlider.value = similarityThreshold;
  thresholdValue.textContent = similarityThreshold.toFixed(2);

  thresholdSlider.addEventListener("input", (e) => {
    similarityThreshold = parseFloat(e.target.value);
    thresholdValue.textContent = similarityThreshold.toFixed(2);

    // 如果已有数据，更新视图
    if (graphData) {
      updateGraph();
    }
  });

  // 重置视图按钮
  document.getElementById("reset-zoom").addEventListener("click", () => {
    svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
  });
}

// 获取数据函数
function fetchData() {
  // 显示加载指示器
  container
    .append("text")
    .attr("x", width / 2)
    .attr("y", height / 2)
    .attr("text-anchor", "middle")
    .text("正在加载数据...")
    .attr("class", "loading-text");

  // 从API获取数据
  fetch(`/api/face-graph?similarity_threshold=${similarityThreshold}`)
    .then((response) => {
      if (!response.ok) {
        throw new Error("网络响应错误");
      }
      return response.json();
    })
    .then((data) => {
      graphData = data;
      renderGraph();
    })
    .catch((error) => {
      console.error("获取数据失败:", error);
      container.select(".loading-text").text("获取数据失败，请刷新页面重试。");
    });
}

// 更新图形，基于新的阈值筛选边
function updateGraph() {
  // 重新获取数据（或者只获取边的数据）
  fetch(`/api/face-graph?similarity_threshold=${similarityThreshold}`)
    .then((response) => response.json())
    .then((data) => {
      graphData = data;
      renderGraph();
    })
    .catch((error) => {
      console.error("更新数据失败:", error);
    });
}

// 渲染图形
function renderGraph() {
  // 清除之前的图形
  container.selectAll("*").remove();

  // 为每个节点添加图案定义
  const defs = svg.select("defs");
  defs.selectAll("*").remove();

  graphData.nodes.forEach((node) => {
    defs
      .append("pattern")
      .attr("id", `image-${node.id}`)
      .attr("width", 1)
      .attr("height", 1)
      .attr("patternContentUnits", "objectBoundingBox")
      .append("image")
      .attr("xlink:href", `data:image/png;base64,${node.image_data}`)
      .attr("width", 1)
      .attr("height", 1)
      .attr("preserveAspectRatio", "xMidYMid slice");
  });

  // 绘制连线
  const links = container
    .append("g")
    .attr("class", "links")
    .selectAll("line")
    .data(graphData.edges)
    .enter()
    .append("line")
    .attr("class", "link")
    .attr("stroke", (d) => {
      // 根据相似度设置颜色，相似度越高颜色越亮
      const hue = 220; // 蓝色
      const saturation = 90; // 饱和度
      const lightness = 30 + d.similarity * 50; // 亮度随相似度增加
      return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    })
    .attr("stroke-width", (d) => {
      // 根据相似度设置线宽，相似度越高线越粗，但整体变细
      return 0.5 + d.similarity * 2;
    });

  // 添加连线上的相似度标签
  const linkLabels = container
    .append("g")
    .attr("class", "link-labels")
    .selectAll("text")
    .data(graphData.edges)
    .enter()
    .append("text")
    .attr("class", "link-label")
    .text((d) => d.similarity.toFixed(2))
    .attr("font-size", "10px")
    .attr("fill", "#333")
    .attr("background", "white")
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "middle")
    .attr("paint-order", "stroke")
    .attr("stroke", "white")
    .attr("stroke-width", "3px")
    .attr("stroke-linecap", "butt")
    .attr("stroke-linejoin", "miter");

  // 绘制节点
  const nodes = container
    .append("g")
    .attr("class", "nodes")
    .selectAll(".node")
    .data(graphData.nodes)
    .enter()
    .append("g")
    .attr("class", "node")
    .call(d3.drag().on("start", dragStarted).on("drag", dragged).on("end", dragEnded));

  // 添加圆形节点，使用图案填充
  nodes
    .append("circle")
    .attr("r", 30)
    .attr("fill", (d) => `url(#image-${d.id})`)
    .attr("stroke", "#1a73e8")
    .attr("stroke-width", 2);

  // 添加名称标签
  nodes.append("text").attr("dy", 45).attr("text-anchor", "middle").attr("fill", "#333");

  // 悬停事件 - 显示向量数据
  nodes.on("mouseover", showVectorInfo).on("mouseout", hideVectorInfo);

  // 创建力模拟
  simulation = d3
    .forceSimulation(graphData.nodes)
    .force(
      "link",
      d3
        .forceLink(graphData.edges)
        .id((d) => d.id)
        .distance((d) => {
          // 相似度越高，距离越近
          return 200 * (1 - d.similarity);
        })
    )
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(width / 2, height / 2).strength(0.1))
    .force("collision", d3.forceCollide().radius(60))
    .force("x", d3.forceX(width / 2).strength(0.05))
    .force("y", d3.forceY(height / 2).strength(0.05))
    .on("tick", ticked);

  // 定位函数
  function ticked() {
    links
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);

    // 更新连线标签位置
    linkLabels
      .attr("x", (d) => (d.source.x + d.target.x) / 2)
      .attr("y", (d) => (d.source.y + d.target.y) / 2);

    nodes.attr("transform", (d) => `translate(${d.x}, ${d.y})`);
  }

  // 拖拽开始
  function dragStarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }

  // 拖拽中
  function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
  }

  // 拖拽结束
  function dragEnded(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }
}

// 显示向量信息
function showVectorInfo(event, d) {
  const vectorInfo = document.getElementById("vector-info");
  const vectorData = document.querySelector(".vector-data");

  // 格式化向量数据展示，只显示前10个元素和后10个元素
  const vector = d.vector;
  let displayVector;

  if (vector.length > 20) {
    const first10 = vector.slice(0, 10);
    const last10 = vector.slice(-10);
    displayVector = [...first10, "...", ...last10];
  } else {
    displayVector = vector;
  }

  // 设置向量数据
  vectorData.innerHTML = `
        <p><strong>向量(${vector.length}维):</strong></p>
        <pre>${JSON.stringify(displayVector, null, 2)}</pre>
    `;

  // 显示向量信息面板
  vectorInfo.classList.remove("hidden");
}

// 隐藏向量信息
function hideVectorInfo() {
  const vectorInfo = document.getElementById("vector-info");
  vectorInfo.classList.add("hidden");
}

// 页面加载完成后初始化
document.addEventListener("DOMContentLoaded", init);

// 窗口大小改变时重绘
window.addEventListener("resize", () => {
  if (svg) {
    // 更新尺寸
    const graphContainer = document.getElementById("graph-container");
    width = graphContainer.clientWidth;
    height = graphContainer.clientHeight;

    svg.attr("width", width).attr("height", height);

    // 如果已有数据，重新渲染
    if (graphData && simulation) {
      simulation.force("center", d3.forceCenter(width / 2, height / 2));
      simulation.alpha(0.3).restart();
    }
  }
});
