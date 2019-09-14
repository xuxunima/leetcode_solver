# 动态规划
动态规划基本思想：将待求解问题分解成若干个子问题，先求解子问题，然后从这些子问题的解得到原问题的解（这部分与分治法相似）。与分治法不同的是，适合于用动态规划求解的问题，经分解得到的子问题往往不是互相独立的。若用分治法来解这类问题，则分解得到的子问题数目太多，有些子问题被重复计算了很多次。如果我们能够保存已解决的子问题的答案，而在需要时再找出已求得的答案，这样就可以避免大量的重复计算，节省时间。通常可以用一个表来记录所有已解的子问题的答案。不管该子问题以后是否被用到，只要它被计算过，就将其结果填入表中。这就是动态规划的基本思路。
 ———————————————— 

## 最长公共子序列问题
>Given two strings text1 and text2, return the length of their longest common subsequence.
A subsequence of a string is a new string generated from the original string with some characters(can be none) deleted without changing the relative order of the remaining characters. (eg, "ace" is a subsequence of "abcde" while "aec" is not). A common subsequence of two strings is a subsequence that is common to both strings.   
If there is no common subsequence, return 0.

假设dp[i][j]表示text1[0:i]和text2[0:j]的最长公共子序列，则其递归推导式为：    
>if text1[i] == text2[j]  
&nbsp;&nbsp;&nbsp;&nbsp;dp[i][j] = dp[i-1][j-1]+1   
else   
&nbsp;&nbsp;&nbsp;&nbsp; dp[i][j] = max(dp[i][j-1], dp[i-1][j])   
>
### 递归
根据上面的递归推导公式，可以写得到递归方法如下：
```
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        return helper(text1, text2, text1.size()-1, text2.size()-1);
    }
    
    int helper(string& text1, string& text2, int i, int j)
    {
        if (i == -1 || j == -1)
            return 0;
        if (text1[i] == text2[j])
        {
            return helper(text1, text2, i-1, j-1) + 1;
        }
        else
            return max(helper(text1, text2, i, j-1), helper(text1, text2, i-1, j));
    }
};
```
但是可以以上递归方法当序列较长时，会出现Time Limit Exceeded，原因是递归调用中出现了很多重复计算，可以采用Memoization技术，把每一次的结果先用一个map保存起来，下次计算时先看map中有没有，有的话就可以直接用了。
```
class Solution {
    unordered_map<string, int>m;
public:
    int longestCommonSubsequence(string text1, string text2) {
        return helper(text1, text2, text1.size()-1, text2.size()-1);
    }
    
    int helper(string& text1, string& text2, int i, int j)
    {
        if (i == -1 || j == -1)
            return 0;
        string str = to_string(i)+"#"+to_string(j);
        if (m.count(str)!=0)
            return m[str];
        int cnt;
        if (text1[i] == text2[j])
        {
            cnt = helper(text1, text2, i-1, j-1) + 1;
        }
        else
            cnt = max(helper(text1, text2, i, j-1), helper(text1, text2, i-1, j));
         m[str] = cnt;
        return cnt;
    }
};

```

### 动态规划
动态规划采用一个表来存储中间结果，以空间换取时间
```buildoutcfg
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        vector<vector<int>>dp(text1.size()+1, vector<int>(text2.size()+1, 0));
        
        for (int i=1;i<=text1.size();i++)
        {
            for (int j=1;j<=text2.size();j++)
            {
                if (text1[i-1] == text2[j-1])
                {
                    dp[i][j] = dp[i-1][j-1] + 1;
                }
                else
                    dp[i][j] = max(dp[i][j-1], dp[i-1][j]);
            }
        }
        return dp[text1.size()][text2.size()];
    }
};
```
把动态规划的路径画出来如下图：  

![LCS路径](https://github.com/xuxunima/leetcode_solver/blob/master/dp/image/LCS.png)


当text1或者text2为空串时，最长公共子序列为0，即第一行和第一列全为0.   
当计算dp[i][j]时，若text1[i-1] == text2[j-1],则dp[i][j]是其斜对角元素加1(如下图中黄色框)，若不相等，则dp[i][j]是其左边和上边两元素的最大值(如下图中红色部分)。  
![LCS](https://github.com/xuxunima/leetcode_solver/blob/master/dp/image/LCS2.PNG)  
  
因此，我们求当前值的时候只需要它的斜对角和上面以及左边三个值，换句话说，求当前列的值时，只需要前一列的信息，所以我们不需要用一个二维数组来存储信息，只需要一个一维数组就够了。  
还存在的一个问题是，因为使用一维数组，所以dp[3][1]已经把dp[3][0]覆盖了，所以需要一个prev变量来保存。
```buildoutcfg
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
       vector<int>dp(text1.size()+1, 0);
        for (int j = 1;j<=text2.size();j++)
        {
            int prev = dp[0];
            for (int i=1;i<=text1.size();i++)
            {
                int temp = dp[i];
                if (text1[i-1] == text2[j-1])
                    dp[i] = prev+1;
                else
                    dp[i] = max(dp[i-1], dp[i]);
                prev = temp;
            }
        }
        return dp[text1.size()];
    }
};
```
(动态规划太强了！)
### 输出所有最长公共子序列
基本思想如下：  
1.如果dp[i][j]对应的text1[i-1]和text2[j-1]相等，则把该字符存入lcs_str，并跳入dp[i-1][j-1]中继续进行判断;  
2.若不相等，则比较dp[i-1][j]和dp[i][j-1]值的大小，跳入大的那一个继续进行判断；   
3.直到i或者j小于等于0为止，倒序输出lcs_str;   
Note:若dp[i-1][j]和dp[i][j-1]的值一样大，说明最长公共子序列有多个，两边都需要回溯(用递归)。
![LCS回溯](https://github.com/xuxunima/leetcode_solver/blob/master/dp/image/LCS3.jpg)
```buildoutcfg
set<string>setOfLSC;
void traceBack(int i, int j, string& lcs_str)
{
	while (i>0 && j>0)
	{
		if (text1[i-1] == text2[j-1])
		{
			lcs_str.push_back(text1[i-1]);
			--i;
			--j;
		}
		else
		{
			if (dp[i-1][j] > dp[i][j-1])
				--i;
			else if (dp[i-1][j] < dp[i][j-1])
				--j;
			else   // 相等的情况
			{
				traceBack(i-1, j, lcs_str);
				traceBack(i, j-1, lcs_str);
				return;
			}
		}
	}
  
	setOfLCS.insert(reverse(lcs_str.begin(), lcs_str.end()));
}

```

## 最长公共子串问题
子串问题和子序列问题的区别是，子串要求字符是连续的，而字序列不需要连续。   
>Given two integer arrays A and B, return the maximum length of an subarray that appears in both arrays.

>Example 1:   
Input:  
A: [1,2,3,2,1]  
B: [3,2,1,4,7]   
Output: 3   
Explanation: 
The repeated subarray with maximum length is [3, 2, 1].   

因此这时候的递推公式如下：
>if (text1[i-1] == text[j-1])   
&nbsp;&nbsp;&nbsp;&nbsp;dp[i][j] = dp[i-1][j-1] + 1  
else   
&nbsp;&nbsp;&nbsp;&nbsp;dp[i][j] = 0   
>
并用一个全局变量res保存最长公共子串。
### 方法一(二维数组dp)
```buildoutcfg
class Solution {
public:
    int findLength(vector<int>& A, vector<int>& B) {
        int s1_len = A.size();
	int s2_len = B.size();
	int res = 0;
	vector<vector<int>>dp(s1_len + 1, vector<int>(s2_len + 1, 0));
	for (int i = 1; i <= s1_len; i++)
	{
		for (int j = 1; j <= s2_len; j++)
		{
			if (A[i - 1] == B[j - 1])
			{
				dp[i][j] = dp[i - 1][j - 1] + 1;
				if (dp[i][j] > res)
					res = dp[i][j];
			}
		}
	}
	return res;
    }
};
```
### 方法二(一维数组dp)
```buildoutcfg
class Solution {
public:
    int findLength(vector<int>& A, vector<int>& B) {
        int A_len = A.size();
        int B_len = B.size();
        vector<int>dp(A_len+1);
        int res = 0;
        for (int j=1;j<=B_len;j++)
        {
            int prev = dp[0];
            for (int i=1;i<=A_len;i++)
            {
                int temp = dp[i];
                if (A[i-1] == B[j-1])
                {
                    dp[i] = prev + 1;
                    if (dp[i] > res)
                        res = dp[i];
                }
                    
                else
                    dp[i] = 0;
                prev = temp;
            }
        }
        return res;
    }
};
```

## 最长递增子序列(Longest Increasing Subsequence)
设dp[i]表示至下标为i处的最长递增子序列，则递推公式为：
>dp[i] = dp[j] + 1 , j < i and A[j] < A[i]  
dp[i] = 1 , else  
>

### 递归方法(memoization)
```buildoutcfg
class Solution {
    int max_v = 0;
    unordered_map<int, int>m;
public:
    int lengthOfLIS(vector<int>& nums) {
        if (nums.size() == 0)
            return 0;
        for (int i=0;i<nums.size();i++)
        {
            helper(nums, i);
        }
       
        return max_v;
    }
    
    int helper(vector<int>& nums, int i)
    {
        if (i == 0)
        {
            max_v = max(max_v, 1);
            return 1;
        }
        if (m.count(i)!=0)
            return m[i];
        int res = 1;
        for (int j=0;j<i;j++)
        {
            if (nums[i] > nums[j])
                res = max(res, helper(nums, j)+1);
        }
        if (max_v < res)
            max_v = res;
        m[i] = res;
        return res;  
    }
};
```
### 动态规划方法
```buildoutcfg
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int len = nums.size();
        vector<int>dp(len, 0);
        int max_v = 0;
        for (int i=0;i<len;i++)
        {
            int res = 1;
            for (int j=0;j<i;j++)
            {
                if (nums[i] > nums[j])
                {
                    res = max(res, dp[j]+1);
                }
            }
            dp[i] = res;
            max_v = max(max_v, res);
        }
        return max_v;
    }
};
```
### 动态规划(Binary search)
>开一个栈，依次读取数组元素x和栈顶元素   
1 如果x > top, 将x入栈   
2 如果x < top, 二分查找栈中第一个大于等于x的数，并用x替换它

```buildoutcfg
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        vector<int>v;
        for (int i=0;i<nums.size();i++)
        {
            if (v.size() == 0 ||nums[i] > v.back())
                v.push_back(nums[i]);
            else
            {
                int low = 0;
                int high = v.size()-1;
                while(low < high)
                {
                    int middle = (low + high) / 2;
                    if (v[middle] < nums[i])
                        low = middle+1;
                    else
                        high = middle;
                }
                v[high] = nums[i];
            }
        }
        return v.size();
    }
};
```

## 正则表达式匹配问题
>Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*'.  
'.' Matches any single character.  
'*' Matches zero or more of the preceding element.  
>

### 递归方法
```buildoutcfg
class Solution {
public:
    bool isMatch(string s, string p) {
       if (p.size() == 0)
            return s.size() == 0;
        bool first_match = (s.size()!=0 && ((s[0] == p[0]) || (p[0] == '.')));
        if (p.size() >= 2 && p[1] == '*')
        {
            return (isMatch(s, p.substr(2))||
                (first_match && isMatch(s.substr(1), p)));
        }
        else
            return first_match && isMatch(s.substr(1), p.substr(1));
    }
};
```

### 动态规划
假设dp[i][j]表示text从下标i到末尾和pattern从下标j到末尾是否匹配  
```buildoutcfg
class Solution {
public:
    bool isMatch(string s, string p) {
        int s_len = s.size();
        int p_len = p.size();
        vector<vector<bool>>dp(s_len+1, vector<bool>(p_len+1));
        dp[s_len][p_len] = true; //空串匹配
        for (int i=s_len;i>=0;i--)
        {
            for (int j=p_len;j>=0;j--)
            {
                if (i == s_len && j == p_len)
                    continue;
                bool first_match = (i<s_len && j<p_len && (s[i] == p[j] || p[j] == '.'));
                if (j+1 < p_len && p[j+1] == '*')
                {
                    dp[i][j] = (dp[i][j+2] || (first_match && dp[i+1][j]));
                }
                else
                    dp[i][j] = first_match && dp[i+1][j+1];
            }
        }
        return dp[0][0];
    }
};
```

## 编辑距离问题(72)
>Given two words word1 and word2, find the minimum number of operations required to convert word1 to word2.   
You have the following 3 operations permitted on a word:  
1.Insert a character  
2.Delete a character   
3.Replace a character
>

>Example 1:   
Input: word1 = "horse", word2 = "ros"   
Output: 3   
Explanation:    
horse -> rorse (replace 'h' with 'r')   
rorse -> rose (remove 'r')   
rose -> ros (remove 'e')
>

### 递归
首先分三种情况考虑：  
第一种，先把 horse 变为 ro ，求出它的最短编辑距离，假如是 x，然后 hosre 变成 ros 的编辑距离就可以是 x + 1。因为 horse 已经变成了 ro，然后我们可以把 ros 的 s 去掉，两个字符串就一样了，也就是再进行一次删除操作，所以加 1。

第二种，先把 hors 变为 ros，求出它的最短编辑距离，假如是 y，然后 hosre 变成 ros 的编辑距离就可以是 y + 1。因为 hors 变成了 ros，然后我们可以把 horse 的 e 去掉，两个字符串就一样了，也就是再进行一次删除操作，所以加 1。

第三种，先把 hors 变为 ro，求出它的最短编辑距离，假如是 z，然后我们再把 e 换成 s，两个字符串就一样了，hosre 变成 ros 的编辑距离就可以是 z + 1。当然，如果是其它的例子，最后一个字符是一样的，比如是 hosrs 和 ros ，此时我们直接取 z 作为编辑距离就可以了。

最后，我们从上边所有可选的编辑距离中，选一个最小的就可以了。  

```buildoutcfg
class Solution {
public:
    int minDistance(string word1, string word2) {
        int len1 = word1.size();
        int len2 = word2.size();
        if (len1 ==0 && len2 == 0)
            return 0;
        if (len1 == 0)
            return len2;
        if (len2 == 0)
            return len1;
        int x = minDistance(word1, word2.substr(0, len2-1))+1;
        int y = minDistance(word1.substr(0, len1-1), word2)+1;
        int z = minDistance(word1.substr(0, len1-1), word2.substr(0, len2-1));
        if (word1[len1-1] != word2[len2-1])
            z++;
        return min(min(x, y), z);
    }
};
```

### 动态规划
假设dp[i][j]表示word1从下标0到i-1和word2从下标0到j-1处的最小编辑距离，则递推公式如下：
>if word1[i-1] == word2[j-1]  
&nbsp;&nbsp;&nbsp;&nbsp;dp[i][j] = min(dp[i-1][j-1], dp[i-1][j]+1, dp[i][j-1]+1)  
else  
&nbsp;&nbsp;&nbsp;&nbsp;dp[i][j] = min(dp[i-1][j-1]+1, dp[i-1][j]+1, dp[i][j-1]+1)

```buildoutcfg
class Solution {
public:
    int minDistance(string word1, string word2) {
        int len1 = word1.size();
        int len2 = word2.size();
        vector<vector<int>>dp(len1+1, vector<int>(len2+1, 0));
        for (int i=0;i<=len1;i++)
            dp[i][0] = i;
        for (int j=0;j<=len2;j++)
            dp[0][j] = j;
        for (int i=1;i<=len1;i++)
        {
            for (int j=1;j<=len2;j++)
            {
                if (word1[i-1] == word2[j-1])
                {
                    dp[i][j] = min(min(dp[i-1][j-1], dp[i-1][j]+1), dp[i][j-1]+1);
                }
                else
                    dp[i][j] = min(min(dp[i-1][j]+1, dp[i][j-1]+1), dp[i-1][j-1]+1);
            }
        }
        return dp[len1][len2];
    }
};
```

同理该问题的空间可以进行优化，只需要一个数组就够了，求dp[i][j]时只需要知道他的斜对角以及上边左边三个数
```buildoutcfg
class Solution {
public:
    int minDistance(string word1, string word2) {
        int len1 = word1.size();
        int len2 = word2.size();
        vector<int>dp(len1+1);
        for (int i=0;i<=len1;i++)
            dp[i] = i;
        for (int j=1;j<=len2;j++)
        {
            int prev = dp[0];
            dp[0] = dp[0]+1;
            for (int i=1;i<=len1;i++)
            {
                int temp = dp[i];
                if (word1[i-1] == word2[j-1])
                    dp[i] = min(min(dp[i], dp[i-1])+1, prev);
                else
                    dp[i] = min(min(dp[i], dp[i-1])+1, prev+1);
                prev = temp;
            }
        }
        return dp[len1];
    }
};
```

## 01背包问题
有n个物品，他们有各自的体积和价值，现在有给定容量的背包，求背包里面装入的物品的最大价值？  

设dp[i][j]表示用前i件物品以及给定背包容量j所得到的的最大价值，则可以分两种情况来讨论：  
1 当前物品的重量weights[i-1]大于背包容量，则该物品不能放入背包，dp[i][j] = dp[i-1][j];     
2 当前物品的重量weights[i-1]小于等于背包容量，则该武平可以选择放入背包或者不放入背包，dp[i][j] = max(dp[i-1][j], dp[i-1][j-weights[i-1]+values[i-1]]);  

### 递归方法
```buildoutcfg
int bag(vector<int>&weights, vector<int>& values, int capacity)
{
	if (capacity == 0)
		return 0;
	return helper(weights, values, capacity, weights.size());
}

int helper(vector<int>&weights, vector<int>& values, int capacity, int n)
{
	if (capacity <= 0 || n <= 0)
		return 0;
	int res = helper(weights, values, capacity, n - 1);
	if (weights[n - 1] <= capacity)
	{
		int a = helper(weights, values, capacity - weights[n - 1], n - 1) + values[n - 1];
		res = max(res, a);
	}
	return res;
}
```
Note：当然也可以用一个二维数组或者哈希表来保存中间值，即memorization技术加快速度，具体代码参考之前的题目。  

### 动态规划（二维数组）
```buildoutcfg
int dp_bag(vector<int>&weights, vector<int>& values, int capacity)
{
	if (capacity <= 0)
		return 0;
	vector<vector<int>>dp(weights.size()+1, vector<int>(capacity + 1, 0));
	for (int i = 1; i <= weights.size(); i++)
	{
		for (int j = 1; j <= capacity; j++)
		{
			if (weights[i-1] >j)
				dp[i][j] = dp[i - 1][j];
			else
				dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i-1]] + values[i-1]);
		}
	}
	return dp[weights.size()][capacity];
}
```

把动态规划表画出来之后可以发现求当前行的值只与上一行的值有关，因此可以只用一个一维数组，但是考虑到当前行内容会覆盖上一行内容，所以要从后往前推
```buildoutcfg
int dp_bag2(vector<int>& weights, vector<int>& values, int capacity)
{
	if (capacity <= 0)
		return 0;
	vector<int>dp(capacity+1);
	for (int i = 1; i <= weights.size(); i++)
	{
		for (int j = capacity; j>=1; j--)
		{
			if (weights[i - 1] > j)
				dp[j] = dp[j];
			else
				dp[j] = max(dp[j], dp[j - weights[i - 1]] + values[i - 1]);
		}
	}
	return dp[capacity];
}
```


## 完全背包问题
完全背包问题和01背包问题的差别在于，01背包的物体不能重复放，即只有两种可能放或者不放，而完全背包同一件物品可以重复放多次。    

### 动态规划（二维数组）
```buildoutcfg
int complete_bag(vector<int>& weights, vector<int>& values, int capacity)
{
	if (capacity <= 0)
		return 0;
	vector<vector<int>>dp(weights.size() + 1, vector<int>(capacity + 1, 0));
	for (int i = 1; i <= weights.size(); i++)
	{
		for (int j = 1; j <= capacity; j++)
		{
			dp[i][j] = dp[i - 1][j];
			for (int k = 1; k <= j / weights[i - 1]; k++)
				dp[i][j] = max(dp[i][j], dp[i - 1][j - weights[i - 1] * k] + values[i - 1] * k);
		}
	}
	return dp[weights.size()][capacity];
}
```

### 动态规划（一位数组）
从后往前递推
```buildoutcfg
int complete_bag2(vector<int>& weights, vector<int>& values, int capacity)
{
	if (capacity <= 0)
		return 0;
	vector<int>dp(capacity + 1);
	for (int i = 1; i <= weights.size(); i++)
	{
		for (int j = capacity; j >=1; j--)
		{
			for (int k = 1; k <= j / weights[i - 1]; k++)
				dp[j] = max(dp[j], dp[j - k * weights[i - 1]] + values[i - 1] * k);
		}
	}
	return dp[capacity];
}
```

### 动态规划（一位数组）
从前往后递推
```buildoutcfg
int complete_bag3(vector<int>& weights, vector<int>& values, int capacity)
{
	if (capacity <= 0)
		return 0;
	vector<int>dp(capacity + 1);
	for (int i = 1; i <= weights.size(); i++)
	{
		for (int j = 1; j <=capacity; j++)
		{
			if (weights[i-1] <= j)
				dp[j] = max(dp[j], dp[j - weights[i - 1]] + values[i - 1]);
		}
	}
	return dp[capacity];
}
```

## 零钱兑换（322）
>给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。   
示例 1:   
输入: coins = [1, 2, 5], amount = 11   
输出: 3    
解释: 11 = 5 + 5 + 1   
示例 2:   
输入: coins = [2], amount = 3   
输出: -1
>
设dp[i]表示总金额为i的最少硬币个数，则dp[i] = min(dp[i], dp[i-coin])(coin<=i)  
```buildoutcfg
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        int max = amount + 1;
        vector<int>dp(amount+1, max);
        dp[0] = 0;
        for (int i=1; i<=amount;i++)
        {
            for (int j=0;j<coins.size();j++)
            {
                if (coins[j] <= i)
                    dp[i] = min(dp[i], dp[i-coins[j]]+1);
            }
        }
        return dp[amount] >= amount+1?-1:dp[amount];
    }
};
```

```buildoutcfg
class Solution {
    unordered_map<int, int>m;
public:
    int coinChange(vector<int>& coins, int amount) {
        if (amount == 0)
            return 0;
        if (m.count(amount)!=0)
            return m[amount];
        int min_coins = amount+1;
        for (int i=0;i<coins.size();i++)
        {
            if (coins[i] <= amount)
            {
                int res = coinChange(coins, amount-coins[i]);
                if (res>=0 && res < min_coins)
                    min_coins = res + 1;
            } 
        }
        m[amount] = min_coins>amount?-1:min_coins;
        return m[amount];
    }
};
```
## 零钱兑换II（518）
>给定不同面额的硬币和一个总金额。写出函数来计算可以凑成总金额的硬币组合数。假设每一种面额的硬币有无限个。    
示例 1:   
输入: amount = 5, coins = [1, 2, 5]   
输出: 4   
解释:  
有四种方式可以凑成总金额:  
5=5  
5=2+2+1   
5=2+1+1+1   
5=1+1+1+1+1
>
设dp[i][j]表示用前i个硬币凑成总金额为j的方法数，则  
dp[i][j] += dp[i-1][j-k*coins[i-1]] (k<=j/coins[i-1])  
```buildoutcfg
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        vector<vector<int>>dp(coins.size()+1, vector<int>(amount+1));
        for (int i=0;i<=coins.size();i++)
            dp[i][0] = 1;
        for (int i=1;i<=coins.size();i++)
        {
            for (int j=1;j<=amount;j++)
            {
                dp[i][j] = dp[i-1][j];
                for (int k=1;k<=j/coins[i-1];k++)
                    dp[i][j] += dp[i-1][j-k*coins[i-1]];
            }
        }
        return dp[coins.size()][amount];
    }
};
```

```buildoutcfg
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        vector<vector<int>>dp(coins.size()+1, vector<int>(amount+1));
        for (int i=0;i<=coins.size();i++)
            dp[i][0] = 1;
        for (int i=1;i<=coins.size();i++)
        {
            for (int j=1;j<=amount;j++)
            {
                dp[i][j] = dp[i-1][j];
                if (j >= coins[i-1])
                    dp[i][j] += dp[i][j-coins[i-1]];
            }
        }
        return dp[coins.size()][amount];
    }
};
```

压缩成一维
```buildoutcfg
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        vector<int>dp(amount+1);
        dp[0] = 1;
        for (int i=1;i<=coins.size();i++)
        {
            for (int j=1;j<=amount;j++)
            {
                if (coins[i-1] <= j)
                    dp[j] += dp[j-coins[i-1]];
            }
        }
        return dp[amount];
    }
};
```


## Reference
[【动态规划】输出所有的最长公共子序列](https://blog.csdn.net/lisonglisonglisong/article/details/41596309)  
[最长递增子序列](https://blog.csdn.net/u013074465/article/details/45442067)
 