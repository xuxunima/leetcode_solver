# 回溯法
回溯法的主要思想是一条路走到底，直到走不通了返回上一个节点走另一条路。回溯法的套路是画图，根据画出来的树形图写代码，最后有需要的话加上剪枝步骤。  
本文档包括的leetcode题目有：  
1.[组合总和（39）](https://leetcode-cn.com/problems/combination-sum/)   
2.[组合总和II（40）](https://leetcode-cn.com/problems/combination-sum-ii/)    
3.[组合总数III(216)](https://leetcode-cn.com/problems/combination-sum-iii/)  
4.[组合总数IV](https://leetcode-cn.com/problems/combination-sum-iv/)
5.[全排列（46）](https://leetcode-cn.com/problems/permutations/)  
6.[全排列II（47）](https://leetcode-cn.com/problems/permutations-ii/)      
7.[组合（77）](https://leetcode-cn.com/problems/combinations/)  
8.[子集（78）](https://leetcode-cn.com/problems/subsets/)  
9.[子集II（90）](https://leetcode-cn.com/problems/subsets-ii/)  

## 组合总数（39）
>给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
candidates 中的数字可以无限制重复被选取。  
说明：  
所有数字（包括 target）都是正整数。
解集不能包含重复的组合。   
示例 1:  
输入: candidates = [2,3,6,7], target = 7,  
所求解集为:  
[  
  [7],  
  [2,2,3],  
]
>
首先画出本题的树状图如下：  
![组合总数](sdsddg)  
从图中可以看到target为0处即为所求的根节点，为了保证最后的结果没有重复，首先可以对原数组进行排序，每次路径只取大于等于改点的值，另一方面，当下一个值大于target时，一定不是正确路径，所以可以直接略去，达到剪枝效果。  
### 回溯法
```buildoutcfg
class Solution {
    vector<vector<int>>res;
    vector<int>path;
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        dfs(candidates, target, 0);
        return res;
    }
    
    void dfs(vector<int>& candidates, int target, int start)
    {
        if (target == 0)
        {
            res.push_back(path);
            return;
        }
        for (int i=start;i<candidates.size() && candidates[i]<=target;i++)
        {
            path.push_back(candidates[i]);
            dfs(candidates, target-candidates[i], i);
            path.pop_back();
        }
    }
};
```

###动态规划
本题还能使用动态规划解决，假设dp[i]表示target为i时的所有组合数，则求dp[i]时只需要在所有dp[i-candidates[j]]数组上加上当前的candidates[j]即可。
```buildoutcfg
class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        if (target <= 0)
            return {{}};
        vector<set<vector<int>>>dp(target+1);
        sort(candidates.begin(), candidates.end());
        for (int i=1;i<=target;i++)
        {
           for (int j=0;j<candidates.size() && candidates[j]<=i;j++)
           {
               if (candidates[j] == i)
                   dp[i].insert({i});
               else
               {
                   for (auto v:dp[i-candidates[j]])
                   {
                       v.push_back(candidates[j]);
                       sort(v.begin(), v.end());
                       if (dp[i].count(v) == 0)
                           dp[i].insert(v);
                   }
               }
           }
        }
        vector<vector<int>>res;
        for (auto v:dp[target])
            res.push_back(v);
        return res;
    }
};
```

## 组合总数II(40)
>给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。   
candidates 中的每个数字在每个组合中只能使用一次。  
说明：  
所有数字（包括目标数）都是正整数。
解集不能包含重复的组合。   
示例 1:  
输入: candidates = [10,1,2,7,6,1,5], target = 8,  
所求解集为:  
[  
　[1, 7],  
 　[1, 2, 5],   
 　[2, 6],   
 　[1, 1, 6]   
]

这题和上一题的区别是上一题中没有重复数字，但是每个数字可以取多次，这一题中candidates中可能有重复数字，但是每个数字只能取一次。  
首先画出本题的树形图(第二个2用2’表示)，从图中可以看到当当前数字已经大于target时可以进行剪枝，并且处于同一深度的相同数字也可以进行剪枝。
![组合总数II]()
```buildoutcfg
class Solution {
    vector<vector<int>>res;
    vector<int>path;
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        if (target <= 0)
            return res;
        sort(candidates.begin(), candidates.end());
        dfs(candidates, target, 0);
        return res;
    }
    
    void dfs(vector<int>& candidates, int target, int start)
    {
        if (target == 0)
        {
            res.push_back(path);
            return;
        }
        for (int i=start;i<candidates.size()&& candidates[i]<=target;i++)
        {
            if (i>start && candidates[i] == candidates[i-1])
                continue;
            path.push_back(candidates[i]);
            dfs(candidates, target-candidates[i], i+1);
            path.pop_back();
        }
    }
};
```
## 组合总数III（216）
>找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。  
说明：   
所有数字都是正整数。
解集不能包含重复的组合.     
示例 1:    
输入: k = 3, n = 7   
输出: [[1,2,4]]
>
### 回溯法
```buildoutcfg
class Solution {
    vector<vector<int>>res;
    vector<int>path;
public:
    vector<vector<int>> combinationSum3(int k, int n) {
        dfs(k, n, 1);
        return res;
    }
    
    void dfs(int k, int n, int start)
    {
        if (n == 0 && path.size() == k)
        {
            res.push_back(path);
            return;
        }
        for (int i=start;i<=9 && i<=n;i++)
        {
            path.push_back(i);
            dfs(k, n-i, i+1);
            path.pop_back();
        }
    }
};
```
## 组合总数IV（377）
>给定一个由正整数组成且不存在重复数字的数组，找出和为给定目标正整数的组合的个数。
示例:   
nums = [1, 2, 3]   
target = 4   
所有可能的组合为：   
(1, 1, 1, 1)   
(1, 1, 2)   
(1, 2, 1)   
(1, 3)    
(2, 1, 1)   
(2, 2)   
(3, 1)   
请注意，顺序不同的序列被视作不同的组合。
因此输出为 7。
>
本题只需要求总数有多少，不需要画出每条路径，所以直接采用动态规划方法就可以了，如果要输出所有路径，则需要回溯法。画出树形图后很容易可以看出dp[4] = dp[1]+dp[2]+dp[3]  
```buildoutcfg
class Solution {
public:
    int combinationSum4(vector<int>& nums, int target) {
        if (target == 0)
            return 0;
        vector<int>dp(target+1);
        dp[0] = 1;
        sort(nums.begin(), nums.end());
        for (int i=1;i<=target;i++)
        {
            for (int j=0;j<nums.size()&&nums[j]<=i;j++)
            {
                if (dp[i] > INT_MAX-dp[i-nums[j]])
                {
                    dp[i] = 0;
                    break;
                }
                else
                    dp[i] += dp[i-nums[j]];
            }
        }
        return dp[target];
    }
};
```


## 全排列（46）
>给定一个没有重复数字的序列，返回其所有可能的全排列。  
示例:  
输入: [1,2,3]  
输出:   
[  
　[1,2,3],  
　[1,3,2],   
　[2,1,3],    
　[2,3,1],   
　[3,1,2],  
　[3,2,1]  
]
>

### 回溯法
用一个bool数组used来记录某个数字是否已经添加到路径
```buildoutcfg
class Solution {
    vector<vector<int>>res;
    vector<int>path;
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<bool>used(nums.size(), false);
        dfs(nums, used);
        return res;
    }
    
    void dfs(vector<int>& nums, vector<bool>& used)
    {
        if (path.size() == nums.size())
        {
            res.push_back(path);
            return;
        }
        for (int i=0;i<nums.size();i++)
        {
            if (!used[i])
            {
                path.push_back(nums[i]);
                used[i] = true;
                dfs(nums, used);
                path.pop_back();
                used[i] = false;
            }
            
        }
    }
};
```
### 递归法
本题还可以采用交换递归的方法
```buildoutcfg
class Solution {
    vector<vector<int>>res;
public:
    vector<vector<int>> permute(vector<int>& nums) {
        if (nums.size() == 0)
            return res;
        helper(nums, 0);
        return res;
    }
    
    void helper(vector<int>& nums, int start)
    {
        if (start == nums.size())
        {
            vector<int>t;
            for (auto x: nums)
                t.push_back(x);
            res.push_back(t);
        }
        for (int i=start;i<nums.size();i++)
        {
            swap(nums, i, start);
            helper(nums, start+1);
            swap(nums, start, i);
        }
    }
    
    void swap(vector<int>& nums, int idx1, int idx2)
    {
        int temp = nums[idx1];
        nums[idx1] = nums[idx2];
        nums[idx2] = temp;
    }
};
```

## 全排列II（47）
>给定一个可包含重复数字的序列，返回所有不重复的全排列。  
示例:  
输入: [1,1,2]   
输出:   
[   
　[1,1,2],   
　[1,2,1],   
　[2,1,1]   
]　　
>
本题与上一题的区别是，上一题中不含重复数字，而本题含有重复数字，需要考虑去重问题。
去重的办法和组合总数II一样，对于同一深度的数进行剪枝。
```buildoutcfg
class Solution {
    vector<int>path;
    vector<vector<int>>res;
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<bool>used(nums.size(), false);
        dfs(nums, used);
        return res;
    }
    
    void dfs(vector<int>& nums, vector<bool>& used)
    {
        if (path.size() == nums.size())
        {
            res.push_back(path);
            return;
        }
        for (int i=0;i<nums.size();i++)
        {
            if (i>0 && nums[i-1] == nums[i] && !used[i-1])
                continue;
            if (!used[i])
            {
                path.push_back(nums[i]);
                used[i] = true;
                dfs(nums, used);
                path.pop_back();
                used[i] = false;
            }
        }
    }
};
```

### 递归法
修改上一题的递归法
```buildoutcfg
class Solution {
    vector<vector<int>>res;
    vector<int>path;
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        helper(nums, 0);
        return res;
    }
    
    void helper(vector<int>& nums, int start)
    {
        if (start == nums.size())
        {
            vector<int>t;
            for (auto x: nums)
            {
                t.push_back(x);
            }
            res.push_back(t);
            return ;
        }
        set<int>s;
        for (int i=start;i<nums.size();i++)
        {
            if (s.count(nums[i]) != 0)
                continue;
            s.insert(nums[i]);
            swap(nums, i, start);
            helper(nums, start+1);
            swap(nums, start, i);
        }
    }
    
    void swap(vector<int>& nums, int idx1, int idx2)
    {
        int temp = nums[idx1];
        nums[idx1] = nums[idx2];
        nums[idx2] = temp;
    }
};
```

## 组合（77）
>给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。   
示例:   
输入: n = 4, k = 2  
输出:   
[   
　[2,4],   
　[3,4],   
　[2,3],   
　[1,2],   
　[1,3],   
　[1,4],   
]
>
```buildoutcfg
class Solution {
    vector<vector<int>>res;
    vector<int>path;
public:
    vector<vector<int>> combine(int n, int k) {
        dfs(n, k, 1);
        return res;
    }
    
    void dfs(int n, int k, int start)
    {
        if (path.size() == k)
        {
            res.push_back(path);
            return;
        }
        for (int i=start;i<=n;i++)
        {
            path.push_back(i);
            dfs(n, k, i+1);
            path.pop_back();
        }
    }
};
```

上面的代码中有一部分可以进行剪枝，比如当k=3，n=4时，从3开始的序列就没必须要遍历了，因为肯定达不到k个数。
```buildoutcfg
class Solution {
    vector<vector<int>>res;
    vector<int>path;
public:
    vector<vector<int>> combine(int n, int k) {
        dfs(n, k, 1);
        return res;
    }
    
    void dfs(int n, int k, int start)
    {
        if (path.size() == k)
        {
            res.push_back(path);
            return;
        }
        for (int i=start;i<=n-k+1+path.size();i++)
        {
            path.push_back(i);
            dfs(n, k, i+1);
            path.pop_back();
        }
    }
};
```

## 子集（78）
>给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。  
说明：解集不能包含重复的子集。   
示例:  
输入: nums = [1,2,3]  
输出:  
[  
　[3],  
　[1],  
　[2],  
　[1,2,3],  
　[1,3],  
　[2,3],   
　[1,2],   
　[]   
]
>
### 回溯法
```buildoutcfg
class Solution {
    vector<vector<int>>res;
    vector<int>path;
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        dfs(nums, 0);
        return res;
    }
    
    void dfs(vector<int>& nums, int start)
    {
        res.push_back(path);
        for (int i=start;i<nums.size();i++)
        {
            path.push_back(nums[i]);
            dfs(nums, i+1);
            path.pop_back();
        }
    }
};
```

### 迭代法
![子集迭代]()
每次在上一层循环的基础上插入新的数字。  
```buildoutcfg
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>>res;
        res.push_back({});
        for (int i=0;i<nums.size();i++)
        {
            vector<vector<int>>temp(res);
            for (auto v: res)
            {
                v.push_back(nums[i]);
                temp.push_back(v);
            }
            res = temp;
        }
        return res;
    }
};
```

### 位掩码
子集问题可以考虑位掩码方法
![位掩码]()
```buildoutcfg
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>>res;
        int len = nums.size();
        int size = 1<<len;
        for (int i=0;i<size;i++)
        {
            vector<int>temp;
            for (int j=0;j<len;j++)
            {
                if (i>>j & 1 == 1)
                    temp.push_back(nums[j]);
            }
            res.push_back(temp);
        }
        return res;
    }
};
```

## 子集II(90)
>给定一个可能包含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。   
说明：解集不能包含重复的子集。  
示例:   
输入: [1,2,2]   
输出:   
[   
　[2],   
　[1],   
　[1,2,2],   
　[2,2],  
　[1,2],  
　[]   
]
>
本题和上一题的区别是本题数组中可能出现重复数字，所以需要去重。画出树形图后发现去重的办法也是对处于同一深度的值剪枝。
### 回溯法
```buildoutcfg
class Solution {
    vector<vector<int>>res;
    vector<int>path;
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        dfs(nums, 0);
        return res;
    }
    
    void dfs(vector<int>& nums, int start)
    {
        res.push_back(path);
        for (int i=start;i<nums.size();i++)
        {
            if (i > start && nums[i] == nums[i-1])
                continue;
            path.push_back(nums[i]);
            dfs(nums, i+1);
            path.pop_back();
        }
    }
};
```
Note：迭代的插入法这里不写了，大致思想和上面一样，只不过对数组排序之后，比如有2个2，可以在上一次循环得到的数组后面插入1个2，插入2个2。

### 位掩码
位掩码方法也要考虑去重问题（只选择连续出现的数字）
```buildoutcfg
class Solution {
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>>res;
        int len = nums.size();
        int size = 1<<len;
        for (int i=0;i<size;i++)
        {
            vector<int>temp;
            bool isvalid = true;
            for (int j=0;j<len;j++)
            {
                if (i>>j & 1 == 1)
                {
                     temp.push_back(nums[j]);
                    if (j>0 && nums[j] == nums[j-1] && (i>>(j-1)&1) == 0)
                    {
                        isvalid = false;
                        break;
                    }
                }
            }
            if (isvalid)
                res.push_back(temp);
        }
        return res;
    }
};
```






