---
title: 迪杰斯特拉算法
date: 2024-02-08 18:06:07
categories: 
  - 算法
tags:
  - Dijkstra
top_img: 'linear-gradient(20deg, #0062be, #925696, #cc426e, #fb0347)'
cover: https://s2.loli.net/2024/02/14/Uyfg3BL9XYASjIP.jpg
copyright_author: dadazhang
copyright_author_href: https://dadazhangn.com
copyright_url: https://dadazhangn.com
copyright_info: 此文章版权归公共所有，欢迎转载，
---


## 初始化
 将起始节点标记为已访问，并将其距离设为 0。将所有其他节点的距离设为无穷大，表示未知距离。将一个空的优先级队列用于存储待处理的节点，并将起始节点放入队列中。

## 迭代
  1、从优先级队列中选择距离最短的节点。
  2、对于该节点的每个邻居节点，计算从起始节点到该邻居节点的距离。如果经过当前节点到达邻居节点的路径距离比已知的距离短，则更新邻居节点的距离，并将其添加到优先级队列中。

## 最终结果
   当优先级队列为空时，所有节点的最短路径都已经计算完成。可以从起始节点到每个节点的距离信息中获取最短路径。

Dijkstra 算法的时间复杂度取决于使用的数据结构。当使用最小堆实现优先级队列时，Dijkstra 算法的时间复杂度为 O((V+E)logV)，其中 V 是节点数量，E 是边数量。


``` python
import heapq

def dijkstra(graph, start):
    # 初始化节点到起始节点的距离为无穷大
    distances = {node: float('inf') for node in graph}
    # 起始节点到自身的距离为 0    
    distances[start] = 0
    
    # 使用优先级队列存储节点和其对应的距离
    priority_queue = [(0, start)]

    while priority_queue:
        # 从优先级队列中取出当前距离最短的节点
        current_distance, current_node = heapq.heappop(priority_queue)

        # 如果当前节点的距离大于已知距离，则跳过
        if current_distance > distances[current_node]:
            continue

        # 遍历当前节点的邻居节点
        for neighbor, weight in graph[current_node].items():
            # 计算从起始节点到邻居节点的距离
            distance = current_distance + weight
            # 如果经过当前节点到达邻居节点的距离比已知的距离短，则更新邻居节点的距离
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                # 将邻居节点加入优先级队列中
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

```
# 示例图
```python
  graph = {
      'A': {'B': 1, 'C': 4},
      'B': {'A': 1, 'C': 2, 'D': 5},
      'C': {'A': 4, 'B': 2, 'D': 1},
      'D': {'B': 5, 'C': 1}
}

  start_node = 'A'
  shortest_distances = dijkstra(graph, start_node)
  print("Shortest distances from node", start_node + ":")
  print(shortest_distances)
```