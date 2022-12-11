#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <span>
#include <array>
#include <random>
#include <stack>
using namespace std;

int ans = 0;
int no_of_chars;
int NUM_STATION = 100;
std::vector<std::vector<int>> memo(1001, std::vector<int>(1001));

int main()
{
    cout << 1 << endl;
    return 0;
}

void PrintPairs(std::span<int> arr)
{
    std::vector<int> v;
    for (int i = 0; i < arr.size(); ++i)
    {
        for (int j = i + 1; j < arr.size(); ++j)
        {
            if (std::abs(arr[i]) == std::abs(arr[j]))
            {
                v.push_back(std::abs(arr[i]));
            }
        }
    }
    if (v.empty())
    {
        return;
    }
    std::sort(v.begin(), v.end());
    for (int i = 0; i < v.size(); ++i)
    {
        std::cout << -v[i] << " ▁ " << v[i];
    }
}

double FindArea(double a, double b, double c)
{
    if (a < 0 || b < 0 || c < 0 || (a + b <= c) || a + c <= b || b + c <= a)
    {
        std::cout << "Not a valid triangle\n";
        std::exit(0);
    }
    double s = (a + b + c) / 2;
    return std::sqrt(s * (s - a) * (s - b) * (s - c));
}

bool IsMajority(std::span<int> a)
{
    std::unordered_map<int, int> mp;
    for (int i = 0; i < a.size(); ++i)
    {
        ++mp[a[i]];
    }
    for (const auto &[key, value] : mp)
    {
        if (value >= a.size() / 2)
        {
            return true;
        }
    }
    return false;
}

int NonDecNums(int n)
{
    std::vector<std::vector<int>> a(n + 1, std::vector<int>(10));
    for (int i = 0; i <= 9; ++i)
    {
        a[0][i] = 1;
    }
    for (int i = 1; i <= n; ++i)
    {
        a[i][9] = 1;
    }
    for (int i = 1; i <= n; ++i)
    {
        for (int j = 8; j >= 0; --j)
        {
            a[i][j] = a[i - 1][j] + a[i][j + 1];
        }
    }
    return a[n][0];
}

void FirstFit(std::span<int> block_size, std::span<int> process_size)
{
    std::vector<int> allocation(process_size.size(), -1);

    for (int i = 0; i < process_size.size(); ++i)
    {
        for (int j = 0; j < block_size.size(); ++j)
        {
            if (block_size[j] >= process_size[i])
            {
                allocation[i] = j;
                block_size[j] -= process_size[i];
                break;
            }
        }
    }

    std::cout << "\nProcess No.\tProcess Size\tBlock no.\n";
    for (int i = 0; i < process_size.size(); ++i)
    {
        std::cout << " " << i + 1 << "\t\t" << process_size[i] << "\t\t";
        if (allocation[i] != -1)
        {
            std::cout << allocation[i] + 1;
        }
        else
        {
            std::cout << "Not Allocated";
        }
        std::cout << std::endl;
    }
}

int FirstNonRepeating(std::span<int> arr)
{
    std::unordered_map<int, int> mp;
    for (int i = 0; i < arr.size(); ++i)
    {
        ++mp[arr[i]];
    }
    for (int i = 0; i < arr.size(); ++i)
    {
        if (mp[arr[i]] == 1)
        {
            return arr[i];
        }
    }
    return -1;
}

bool DistributingBalls(int k, std::string_view str)
{
    std::vector<int> a(26);
    for (int i = 0; i < str.size(); ++i)
    {
        ++a[str[i] - 'a'];
    }
    for (int i = 0; i < 26; ++i)
    {
        if (a[i] > k)
        {
            return false;
        }
    }
    return true;
}

int CountOfWays(int n)
{
    int count = (n + 1) * (n + 2) / 2;
    return count;
}

int MaxSum(std::span<int> arr)
{
    int cum_sum = 0;
    for (int i = 0; i < arr.size(); ++i)
    {
        cum_sum += arr[i];
    }
    int curr_val = 0;
    for (int i = 0; i < arr.size(); ++i)
    {
        curr_val += i * arr[i];
    }
    int res = curr_val;
    for (int i = 1; i < arr.size(); ++i)
    {
        int next_val = curr_val - (cum_sum - arr[i - 1]) + arr[i - 1] * (arr.size() - 1);
        curr_val = next_val;
        res = std::max(res, next_val);
    }
    return res;
}

int CalcAngle(double h, double m)
{
    if (h < 0 || m < 0 || h > 12 || m > 60)
    {
        std::cout << "Wrong input" << std::endl;
    }
    if (h == 12)
    {
        h = 0;
    }
    if (m == 60)
    {
        m = 0;
    }
    int hour_angle = 0.5 * (h * 60 + m);
    int minute_angle = 6 * m;
    int angle = std::abs(hour_angle - minute_angle);
    angle = std::min(360 - angle, angle);
    return angle;
}

int Smallest(int x, int y, int z)
{
    if (!(y / x))
    {
        return (!(y / z)) ? y : z;
    }
    return (!(x / z)) ? x : z;
}

char FindExtraCharcter(std::string_view str_a, std::string_view str_b)
{
    int res = 0;
    for (int i = 0; i < str_a.length(); ++i)
    {
        res ^= str_a[i];
    }
    for (int i = 0; i < str_b.length(); ++i)
    {
        res ^= str_b[i];
    }
    return static_cast<char>(res);
}

void Recaman(int n)
{
    if (n <= 0)
    {
        return;
    }
    std::printf("%d, ", 0);
    std::unordered_set<int> s;
    s.insert(0);
    int prev = 0;
    for (int i = 1; i < n; ++i)
    {
        int curr = prev - i;
        if (curr < 0 || s.find(curr) != s.end())
        {
            curr = prev + i;
        }
        s.insert(curr);
        std::printf("%d, ", curr);
        prev = curr;
    }
}

int FindMinimumAngle(std::span<int> arr)
{
    int l = 0, sum = 0, ans = 360;
    for (int i = 0; i < arr.size(); ++i)
    {
        sum += arr[i];
        while (sum >= 180)
        {
            ans = std::min(ans, 2 * std::abs(180 - sum));
            sum -= arr[l];
            ++l;
        }
        ans = std::min(ans, 2 * std::abs(180 - sum));
    }
    return ans;
}

void CountSubsequence(std::string_view str)
{
    int cnt_g = 0, cnt_f = 0, result = 0, c = 0;
    for (int i = 0; i < str.length(); i++)
    {
        switch (str[i])
        {
        case 'G':
            ++cnt_g;
            result += c;
            break;
        case 'F':
            ++cnt_f;
            c += cnt_g;
            break;
        default:
            continue;
        }
    }
    std::cout << result << std::endl;
}

int MaxDiff(std::span<int> arr)
{
    std::unordered_map<int, int> freq;
    for (int i = 0; i < arr.size(); i++)
    {
        freq[arr[i]]++;
    }
    int ans = 0;
    for (int i = 0; i < arr.size(); i++)
    {
        for (int j = 0; j < arr.size(); j++)
        {
            if (freq[arr[i]] > freq[arr[j]] && arr[i] > arr[j])
            {
                ans = std::max(ans, freq[arr[i]] - freq[arr[j]]);
            }
            else if (freq[arr[i]] < freq[arr[j]] && arr[i] < arr[j])
            {
                ans = std::max(ans, freq[arr[j]] - freq[arr[i]]);
            }
        }
    }
    return ans;
}

bool AreEqual(std::span<int> arr1, std::span<int> arr2)
{
    if (arr1.size() != arr2.size())
    {
        return false;
    }
    std::sort(arr1.begin(), arr1.end());
    std::sort(arr2.begin(), arr2.end());
    for (int i = 0; i < arr1.size(); i++)
    {
        if (arr1[i] != arr2[i])
        {
            return false;
        }
    }
    return true;
}

int CountFriendsPairings(int n)
{
    int a = 1, b = 2, c = 0;
    if (n <= 2)
    {
        return n;
    }
    for (int i = 3; i <= n; i++)
    {
        c = b + (i - 1) * a;
        a = b;
        b = c;
    }
    return c;
}

int PowerOfPiNnFactorial(int n, int p)
{
    int ans = 0;
    int temp = p;
    while (temp <= n)
    {
        ans += n / temp;
        temp = temp * p;
    }
    return ans;
}

int FindLength(std::string_view str)
{
    int max_len = 0;
    for (int i = 0; i < str.length(); i++)
    {
        for (int j = i + 1; j < str.length(); j += 2)
        {
            int length = j - i + 1;
            int left_sum = 0, right_sum = 0;
            for (int k = 0; k < length / 2; k++)
            {
                left_sum += (str[i + k] - '0');
                right_sum += (str[i + k + length / 2] - '0');
            }
            if (left_sum == right_sum && max_len < length)
            {
                max_len = length;
            }
        }
    }
    return max_len;
}

int Search(std::span<int> arr, int x)
{
    for (int i = 0; i < arr.size(); i++)
    {
        if (arr[i] == x)
        {
            return i;
        }
    }
    return -1;
}

void SubArray(std::span<int> arr)
{
    for (int i = 0; i < arr.size(); i++)
    {
        for (int j = i; j < arr.size(); j++)
        {
            for (int k = i; k <= j; k++)
            {
                std::cout << arr[k] << " ▁ ";
            }
            std::cout << std::endl;
        }
    }
}

bool IsSubSeqDivisible(std::string_view str)
{
    std::vector<std::vector<int>> dp(str.length() + 1, std::vector<int>(10));
    std::vector<int> arr(str.length() + 1);
    for (int i = 1; i <= str.length(); i++)
    {
        arr[i] = str[i - 1] - '0';
    }
    for (int i = 1; i <= str.length(); i++)
    {
        dp[i][arr[i] % 8] = 1;
        for (int j = 0; j < 8; j++)
        {
            if (dp[i - 1][j] > dp[i][(j * 10 + arr[i]) % 8])
            {
                dp[i][(j * 10 + arr[i]) % 8] = dp[i - 1][j];
            }
            if (dp[i - 1][j] > dp[i][j])
            {
                dp[i][j] = dp[i - 1][j];
            }
        }
    }
    for (int i = 1; i <= str.length(); i++)
    {
        if (dp[i][0] == 1)
        {
            return true;
        }
    }
    return false;
}

void Solve(int i, int par, std::span<int> a, int k, int current_ans)
{
    if (par > k)
    {
        return;
    }
    if (par == k && i == a.size() - 1)
    {
        ans = std::min(ans, current_ans);
        return;
    }
    for (int j = i + 1; j < a.size(); j++)
    {
        Solve(j, par + 1, a, k, current_ans + (a[j] - a[i + 1]) * (a[j] - a[i + 1]));
    }
}

double SquareRoot(double n)
{
    double x = n;
    double y = 1;
    double e = 0.000001;
    while (x - y > e)
    {
        x = (x + y) / 2;
        y = n / x;
    }
    return x;
}

int CountPairs(int n)
{
    int k = n;
    int i_min = 1;
    int ans = 0;
    while (i_min <= n)
    {
        int i_max = n / k;
        ans += k * (i_max - i_min + 1);
        i_min = i_max + 1;
        k = n / i_min;
    }
    return ans;
}

int MultiplyWith3Point5(int x)
{
    return (x << 1) + x + (x >> 1);
}

bool PrevPermutation(std::string &str)
{
    int n = str.length() - 1;
    int i = n;
    while (i > 0 && str[i - 1] <= str[i])
    {
        i--;
    }
    if (i <= 0)
    {
        return false;
    }
    int j = i - 1;
    while (j + 1 <= n && str[j + 1] <= str[i - 1])
    {
        j++;
    }
    std::swap(str[i - 1], str[j]);
    std::reverse(str.begin() + i, str.end());
    return true;
}

double PolygonArea(std::span<double> x, std::span<double> y)
{
    double area = 0.0;
    int j = x.size() - 1;
    for (int i = 0; i < x.size(); i++)
    {
        area += (x[j] + x[i]) * (y[j] - y[i]);
        j = i;
    }
    return std::abs(area / 2.0);
}

int Equilibrium(std::span<int> arr)
{
    for (int i = 0; i < arr.size(); ++i)
    {
        int left_sum = 0;
        for (int j = 0; j < i; j++)
        {
            left_sum += arr[j];
        }
        int right_sum = 0;
        for (int j = i + 1; j < arr.size(); j++)
        {
            right_sum += arr[j];
        }
        if (left_sum == right_sum)
        {
            return i;
        }
    }
    return -1;
}

int ChordCnt(int a)
{
    int n = 2 * a;
    std::vector<int> dp_array(n + 1);
    dp_array[0] = 1;
    dp_array[2] = 1;
    for (int i = 4; i <= n; i += 2)
    {
        for (int j = 0; j < i - 1; j += 2)
        {
            dp_array[i] += (dp_array[j] * dp_array[i - 2 - j]);
        }
    }
    return dp_array[n];
}

double Compute(double a, double b)
{
    double am, gm, hm;
    am = (a + b) / 2;
    gm = sqrt(a * b);
    hm = (gm * gm) / am;
    return hm;
}

void FindMajority(std::span<int> arr)
{
    int max_count = 0;
    int index = -1;
    for (int i = 0; i < arr.size(); i++)
    {
        int count = 0;
        for (int j = 0; j < arr.size(); j++)
        {
            if (arr[i] == arr[j])
                count++;
        }
        if (count > max_count)
        {
            max_count = count;
            index = i;
        }
    }
    if (max_count > arr.size() / 2)
        std::cout << arr[index] << std::endl;
    else
        std::cout << "No Majority Element" << std::endl;
}

bool IsPerfectSquare(int n)
{
    for (int sum = 0, i = 1; sum < n; i += 2)
    {
        sum += i;
        if (sum == n)
            return true;
    }
    return false;
}

std::string findSubString(std::string_view str, std::string_view pat)
{
    int len1 = str.length();
    int len2 = pat.length();
    if (len1 < len2)
    {
        cout << " No ▁ such ▁ window ▁ exists ";
        return " ";
    }
    std::vector<int> hash_pat(no_of_chars);
    std::vector<int> hash_str(no_of_chars);
    for (int i = 0; i < len2; i++)
        hash_pat[pat[i]]++;
    int start = 0, start_index = -1, min_len = std::numeric_limits<int>::max();
    int count = 0;
    for (int j = 0; j < len1; j++)
    {
        hash_str[str[j]]++;
        if (hash_pat[str[j]] != 0 && hash_str[str[j]] <= hash_pat[str[j]])
            count++;
        if (count == len2)
        {
            while (hash_str[str[start]] > hash_pat[str[start]] || hash_pat[str[start]] == 0)
            {
                if (hash_str[str[start]] > hash_pat[str[start]])
                    hash_str[str[start]]--;
                start++;
            }
            int len_window = j - start + 1;
            if (min_len > len_window)
            {
                min_len = len_window;
                start_index = start;
            }
        }
    }
    if (start_index == -1)
    {
        cout << " No ▁ such ▁ window ▁ exists ";
        return " ";
    }
    return std::string(str.substr(start_index, min_len));
}

int CountDivisibleSubseq(std::string_view str, int n)
{
    int len = str.length();
    std::vector<std::vector<int>> dp(len, std::vector<int>(n));
    for (auto &row : dp)
        std::fill(row.begin(), row.end(), 0);
    dp[0][(str[0] - '0') % n]++;
    for (int i = 1; i < len; i++)
    {
        dp[i][(str[i] - '0') % n]++;
        for (int j = 0; j < n; j++)
        {
            dp[i][j] += dp[i - 1][j];
            dp[i][(j * 10 + (str[i] - '0')) % n] += dp[i - 1][j];
        }
    }
    return dp[len - 1][0];
}

void Pairs(std::span<int> arr, int k)
{
    int smallest = std::numeric_limits<int>::max();
    int count = 0;
    for (int i = 0; i < arr.size(); i++)
    {
        for (int j = i + 1; j < arr.size(); j++)
        {
            if (std::abs(arr[i] + arr[j] - k) < smallest)
            {
                smallest = std::abs(arr[i] + arr[j] - k);
                count = 1;
            }
            else if (std::abs(arr[i] + arr[j] - k) == smallest)
            {
                count++;
            }
        }
    }
    std::cout << "Minimal value = " << smallest << "\n";
    std::cout << "Total pairs = " << count << "\n";
}

int NearestSmallerEqFib(int n)
{
    if (n == 0 || n == 1)
    {
        return n;
    }
    int f1 = 0, f2 = 1, f3 = 1;
    while (f3 <= n)
    {
        f1 = f2;
        f2 = f3;
        f3 = f1 + f2;
    }
    return f2;
}

void Shuffle(std::span<int> card)
{
    std::random_device rd;
    std::mt19937 g(rd());
    for (int i = 0; i < card.size(); i++)
    {
        std::uniform_int_distribution<int> d(i, 51);
        int r = d(g);
        std::swap(card[i], card[r]);
    }
}

int MaxLen(std::span<int> arr, int n)
{
    std::unordered_map<int, int> presum;
    int sum = 0;
    int max_len = 0;
    for (int i = 0; i < n; i++)
    {
        sum += arr[i];
        if (arr[i] == 0 && max_len == 0)
        {
            max_len = 1;
        }
        if (sum == 0)
        {
            max_len = i + 1;
        }
        if (presum.find(sum) != presum.end())
        {
            max_len = std::max(max_len, i - presum[sum]);
        }
        else
        {
            presum[sum] = i;
        }
    }
    return max_len;
}

int MaxLength(std::string_view s, int n)
{
    std::vector<std::vector<int>> dp(n, std::vector<int>(n));
    for (int i = 0; i < n - 1; i++)
    {
        if (s[i] == '(' && s[i + 1] == ')')
        {
            dp[i][i + 1] = 2;
        }
    }
    for (int l = 2; l < n; l++)
    {
        for (int i = 0, j = l; j < n; i++, j++)
        {
            if (s[i] == '(' && s[j] == ')')
            {
                dp[i][j] = 2 + dp[i + 1][j - 1];
            }
            for (int k = i; k < j; k++)
            {
                dp[i][j] = std::max(dp[i][j], dp[i][k] + dp[k + 1][j]);
            }
        }
    }
    return dp[0][n - 1];
}

int SummingSeries(int n)
{
    return std::pow(n, 2);
}

bool IsSubsetSum(std::span<int> arr, int sum)
{
    std::vector<std::vector<bool>> subset(2, std::vector<bool>(sum + 1));
    for (int i = 0; i <= arr.size(); i++)
    {
        for (int j = 0; j <= sum; j++)
        {
            if (j == 0)
            {
                subset[i % 2][j] = true;
            }
            else if (i == 0)
            {
                subset[i % 2][j] = false;
            }
            else if (arr[i - 1] <= j)
            {
                subset[i % 2][j] = subset[(i + 1) % 2][j - arr[i - 1]] || subset[(i + 1) % 2][j];
            }
            else
            {
                subset[i % 2][j] = subset[(i + 1) % 2][j];
            }
        }
    }
    return subset[arr.size() % 2][sum];
}

int FindExtra(std::span<int> arr1, std::span<int> arr2)
{
    for (int i = 0; i < arr1.size(); i++)
    {
        if (arr1[i] != arr2[i])
        {
            return i;
        }
    }
    return arr1.size();
}

int NumberOfPermWithKInversion(int n, int k)
{
    if (n == 0)
    {
        return 0;
    }
    if (k == 0)
    {
        return 1;
    }
    if (memo[n][k] != 0)
    {
        return memo[n][k];
    }
    int sum = 0;
    for (int i = 0; i <= k; i++)
    {
        if (i <= n - 1)
        {
            sum += NumberOfPermWithKInversion(n - 1, k - i);
        }
    }
    memo[n][k] = sum;
    return sum;
}

int RemoveConsecutiveSame(std::span<std::string> v)
{
    std::stack<std::string> st;
    for (int i = 0; i < v.size(); i++)
    {
        if (st.empty())
        {
            st.push(v[i]);
        }
        else
        {
            std::string str = st.top();
            if (str.compare(v[i]) == 0)
            {
                st.pop();
            }
            else
            {
                st.push(v[i]);
            }
        }
    }
    return st.size();
}

void Find(std::span<std::string> list1, std::span<std::string> list2)
{
    std::vector<std::string> res;
    int max_possible_sum = list1.size() + list2.size() - 2;
    for (int sum = 0; sum <= max_possible_sum; sum++)
    {
        for (int i = 0; i <= sum; i++)
        {
            if (i < list1.size() && (sum - i) < list2.size() && list1[i] == list2[sum - i])
            {
                res.push_back(list1[i]);
            }
        }
        if (res.size() > 0)
        {
            break;
        }
    }
    for (int i = 0; i < res.size(); i++)
    {
        std::cout << res[i] << " ▁ ";
    }
}

int FirstNonRepeating(std::span<int> arr)
{
    for (int i = 0; i < arr.size(); i++)
    {
        int j;
        for (j = 0; j < arr.size(); j++)
        {
            if (i != j && arr[i] == arr[j])
            {
                break;
            }
        }
        if (j == arr.size())
        {
            return arr[i];
        }
    }
    return -1;
}

int Search(std::span<int> arr, int x)
{
    for (int i = 0; i < arr.size(); i++)
    {
        if (arr[i] == x)
        {
            return i;
        }
    }
    return -1;
}

void MiddleSum(std::span<std::vector<int>> mat, int n)
{
    int row_sum = 0;
    int col_sum = 0;
    for (int i = 0; i < n; i++)
    {
        row_sum += mat[n / 2][i];
    }
    std::cout << "Sum of middle row = " << row_sum << std::endl;
    for (int i = 0; i < n; i++)
    {
        col_sum += mat[i][n / 2];
    }
    std::cout << "Sum of middle column = " << col_sum;
}

int ModuloMultiplication(int a, int b, int mod)
{
    int res = 0;
    a %= mod;
    while (b)
    {
        if (b & 1)
        {
            res = (res + a) % mod;
        }
        a = (2 * a) % mod;
        b >>= 1;
    }
    return res;
}

void PythagoreanTriplets(int limit)
{
    int a, b, c = 0;
    int m = 2;
    while (c < limit)
    {
        for (int n = 1; n < m; ++n)
        {
            a = m * m - n * n;
            b = 2 * m * n;
            c = m * m + n * n;
            if (c > limit)
            {
                break;
            }
            std::cout << a << " " << b << " " << c << std::endl;
        }
        m++;
    }
}

int FindSum(int n)
{
    return n * (n + 1) * (n + 2) * (3 * n + 1) / 24;
}

int Fib(int n)
{
    std::vector<int> f(n + 1);
    if (n == 0)
        return 0;
    if (n == 1 || n == 2)
        return (f[n] = 1);
    if (f[n])
        return f[n];
    int k = (n & 1) ? (n + 1) / 2 : n / 2;
    f[n] = (n & 1) ? (Fib(k) * Fib(k) + Fib(k - 1) * Fib(k - 1)) : (2 * Fib(k - 1) + Fib(k)) * Fib(k);
    return f[n];
}

int LargestPower(int n, int p)
{
    int x = 0;
    while (n)
    {
        n /= p;
        x += n;
    }
    return x;
}

bool IsIdentity(std::span<std::vector<int>> mat, int n)
{
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            if (row == col && mat[row][col] != 1)
                return false;
            else if (row != col && mat[row][col] != 0)
                return false;
        }
    }
    return true;
}

int MinCells(std::span<std::vector<int>> mat, int m, int n)
{
    std::vector<std::vector<int>> dp(m, std::vector<int>(n, std::numeric_limits<int>::max()));
    dp[0][0] = 1;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (dp[i][j] != std::numeric_limits<int>::max() && (j + mat[i][j]) < n && (dp[i][j] + 1) < dp[i][j + mat[i][j]])
            {
                dp[i][j + mat[i][j]] = dp[i][j] + 1;
            }
            if (dp[i][j] != std::numeric_limits<int>::max() && (i + mat[i][j]) < m && (dp[i][j] + 1) < dp[i + mat[i][j]][j])
            {
                dp[i + mat[i][j]][j] = dp[i][j] + 1;
            }
        }
    }
    if (dp[m - 1][n - 1] != std::numeric_limits<int>::max())
    {
        return dp[m - 1][n - 1];
    }
    return -1;
}

void ArrangeString(std::string_view str, int x, int y)
{
    int count_0 = 0;
    int count_1 = 0;
    int len = str.length();
    for (int i = 0; i < len; i++)
    {
        if (str[i] == '0')
            count_0++;
        else
            count_1++;
    }
    while (count_0 > 0 || count_1 > 0)
    {
        for (int j = 0; j < x && count_0 > 0; j++)
        {
            if (count_0 > 0)
            {
                std::cout << "0";
                count_0--;
            }
        }
        for (int j = 0; j < y && count_1 > 0; j++)
        {
            if (count_1 > 0)
            {
                std::cout << "1";
                count_1--;
            }
        }
    }
}

int GetLevensteinDistance(std::string_view input)
{
    std::string rev_input(input.rbegin(), input.rend());
    int n = input.size();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(n + 1, -1));
    for (int i = 0; i <= n; ++i)
    {
        dp[0][i] = i;
        dp[i][0] = i;
    }
    for (int i = 1; i <= n; ++i)
    {
        for (int j = 1; j <= n; ++j)
        {
            if (input[i - 1] == rev_input[j - 1])
            {
                dp[i][j] = dp[i - 1][j - 1];
            }
            else
            {
                dp[i][j] = 1 + std::min({dp[i - 1][j], dp[i][j - 1]});
            }
        }
    }
    int result = std::numeric_limits<int>::max();
    for (int i = n, j = 0; i >= 0; --i, ++j)
    {
        result = std::min(result, dp[i][j]);
        if (i < n)
        {
            result = std::min(result, dp[i + 1][j]);
        }
        if (i > 0)
        {
            result = std::min(result, dp[i - 1][j]);
        }
    }
    return result;
}

int CarAssembly(std::span<std::vector<int>> a, std::span<std::vector<int>> t,
                std::span<int> e, std::span<int> x)
{
    std::vector<int> t1(NUM_STATION), t2(NUM_STATION);
    t1[0] = e[0] + a[0][0];
    t2[0] = e[1] + a[1][0];
    for (int i = 1; i < NUM_STATION; ++i)
    {
        t1[i] = std::min(t1[i - 1] + a[0][i], t2[i - 1] + t[1][i] + a[0][i]);
        t2[i] = std::min(t2[i - 1] + a[1][i], t1[i - 1] + t[0][i] + a[1][i]);
    }
    return std::min(t1[NUM_STATION - 1] + x[0], t2[NUM_STATION - 1] + x[1]);
}

int GCD(int a, int b)
{
    if (a == 0)
    {
        return b;
    }
    return GCD(b % a, a);
}
void FindTriplets(std::span<int> arr)
{
    bool found = false;
    std::sort(arr.begin(), arr.end());
    for (int i = 0; i < arr.size() - 1; i++)
    {
        int l = i + 1;
        int r = arr.size() - 1;
        int x = arr[i];
        while (l < r)
        {
            if (x + arr[l] + arr[r] == 0)
            {
                std::printf(" % d ▁ % d ▁ % d \n ", x, arr[l], arr[r]);
                l++;
                r--;
                found = true;
            }
            else if (x + arr[l] + arr[r] < 0)
            {
                l++;
            }
            else
            {
                r--;
            }
        }
    }
    if (found == false)
    {
        std::cout << " No Triplet Found" << std::endl;
    }
}
int GetOddOccurrence(std::span<int> ar)
{
    int res = 0;
    for (int i = 0; i < ar.size(); i++)
    {
        res = res ^ ar[i];
    }
    return res;
}
int MaxTasks(std::span<int> high, std::span<int> low , int n)
{
    if (n <= 0)
    {
        return 0;
    }
    return std::max(high[n - 1] + MaxTasks(high, low, (n - 2)), low[n - 1] + MaxTasks(high, low, (n - 1)));
}
void PrintOtherSides(int n)
{
    if (n & 1)
    {
        if (n == 1)
        {
            cout << -1 << endl;
        }
        else
        {
            int b = (n * n - 1) / 2;
            int c = (n * n + 1) / 2;
            cout << "b = " << b << " , ▁ c = " << c << endl;
        }
    }
    else
    {
        if (n == 2)
        {
            cout << -1 << endl;
        }
        else
        {
            int b = n * n / 4 - 1;
            int c = n * n / 4 + 1;
            cout << "b = " << b << " , ▁ c = " << c << endl;
        }
    }
}
void PrintUnsorted(std::span<int> arr)
{
    int s = 0, e = arr.size() - 1;
    for (s = 0; s < arr.size(); s++)
    {
        if (arr[s] > arr[s + 1])
        {
            break;
        }
    }
    if (s == arr.size() - 1)
    {
        cout << "The ▁ complete array is sorted";
        return;
    }
    for (e = arr.size() - 1; e > 0; e--)
    {
        if (arr[e] < arr[e - 1])
        {
            break;
        }
    }
    for (int i = 0; i < s; i++)
    {
        if (arr[i] > min)
        {
            s = i;
            break;
        }
    }
    for (int i = n - 1; i >= e + 1; i--)
    {
        if (arr[i] < max)
        {
            e = i;
            break;
        }
    }
    std::cout << "The ▁ unsorted subarray " << std::endl
              << "sorted ▁ between " << s << " and " << e;
    return;
}
int Circle(int x1, int y1, int x2, int y2, int r1, int r2)
{
    int distSq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    int radSumSq = (r1 + r2) * (r1 + r2);
    if (distSq == radSumSq)
        return 1;
    else if (distSq > radSumSq)
        return -1;
    else
        return 0;
}
void CountDistinct(std::span<int> arr, int k, int n)
{
    std::map<int, int> hm;
    int dist_count = 0;
    for (int i = 0; i < k; i++)
    {
        if (hm[arr[i]] == 0)
        {
            dist_count++;
        }
        hm[arr[i]] += 1;
    }
    std::cout << dist_count << std::endl;
    for (int i = k; i < n; i++)
    {
        if (hm[arr[i - k]] == 1)
        {
            dist_count--;
        }
        hm[arr[i - k]] -= 1;
        if (hm[arr[i]] == 0)
        {
            dist_count++;
        }
        hm[arr[i]] += 1;
    }
    std::cout << dist_count << std::endl;
}
}
double FindArea(double a)
{
    double area = (sqrt(5 * (5 + 2 * (sqrt(5)))) * a * a);
    return area;
}
int CountRotationsDivBy8(std::string_view n)
{
    int len = n.length();
    int count = 0;
    if (len == 1)
    {
        int one_digit = n[0] - '0';
        if (one_digit % 8 == 0)
        {
            return 1;
        }
        return 0;
    }
    if (len == 2)
    {
        int first = (n[0] - '0') * 10 + (n[1] - '0');
        if (first % 8 == 0)
            count++;
    }
    int three_digit = (n[len - 2] - '0') * 100 + (n[i + 1] - '0') * 10 + (n[i + 2] - '0');
    if (three_digit % 8 == 0)
    {
        count++;
    }
    three_digit = (n[len - 2] - '0') * 100 + (n[len - 1] - '0') * 10 + (n[0] - '0');
    if (three_digit % 8 == 0)
    {
        count++;
    }
    return count;
}
