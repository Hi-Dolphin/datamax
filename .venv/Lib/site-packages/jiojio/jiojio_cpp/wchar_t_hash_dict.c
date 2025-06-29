#include "wchar_t_hash_dict.h"

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif


DictHashNode **dict_hash_table_init(int hashTableMaxSize)
{
    DictHashNode **dictHashTable = (DictHashNode **)malloc(sizeof(DictHashNode *) * hashTableMaxSize);
    memset(dictHashTable, 0, sizeof(DictHashNode *) * hashTableMaxSize);

    return dictHashTable;
}

// string hash function
unsigned int dict_hash_table_hash_str(const wchar_t *skey)
{
    const signed char *p = (const signed char *)skey;
    unsigned int h = *p;
    int skeyLength = wcslen(skey) * sizeof(wchar_t);

    for (int i = 0; i < skeyLength; i++)
    {
        if (*(p + i) != '\0')
        {
            h = (h << 5) - h + *(p + i);
        }
    }

    return h;
}

// insert key-value into hash table
void dict_hash_table_insert(
    DictHashNode **dictHashTable, const wchar_t *skey, int nvalue, int hashTableMaxSize)
{
    unsigned int pos = dict_hash_table_hash_str(skey) % hashTableMaxSize;
    DictHashNode *pHead = *(dictHashTable + pos);

    while (pHead)
    {
        if (wcscmp(pHead->sKey, skey) == 0)
        {
            printf("%ls already exists!\n", skey);
            return;
        }
        pHead = pHead->pNext;
    }

    DictHashNode *pNewNode = (DictHashNode *)malloc(sizeof(DictHashNode));
    memset(pNewNode, 0, sizeof(DictHashNode));
    pNewNode->sKey = (wchar_t *)malloc(sizeof(wchar_t) * (wcslen(skey) + 1));
    wcscpy(pNewNode->sKey, skey);
    pNewNode->nValue = nvalue;

    // hash 同值链表
    pNewNode->pNext = *(dictHashTable + pos); // 原值附在新值后面
    *(dictHashTable + pos) = pNewNode;        // 新值接在链表头部
}

// remove key-value frome the hash table
void dict_hash_table_remove(DictHashNode **dictHashTable, const wchar_t *skey, int hashTableMaxSize)
{
    unsigned int pos = dict_hash_table_hash_str(skey) % hashTableMaxSize;
    DictHashNode *temp = *(dictHashTable + pos);
    if (temp)
    {
        DictHashNode *pHead = temp;
        DictHashNode *pLast = NULL;
        DictHashNode *pRemove = NULL;
        while (pHead)
        {
            if (wcscmp(skey, pHead->sKey) == 0)
            {
                pRemove = pHead;
                break;
            }
            pLast = pHead;
            pHead = pHead->pNext;
        }

        if (pRemove)
        {
            if (pLast)
                pLast->pNext = pRemove->pNext;
            else
                temp = NULL;

            free(pRemove->sKey);
            free(pRemove);
        }
    }
}

// lookup a key in the hash table
// -1 means failed to lookup and find, >= 0 means success and the value of the key.
// hash_table_hash_str costs 36% of total lookup time. others costs 64%.
int dict_hash_table_lookup(DictHashNode **dictHashTable, const wchar_t *skey, int hashTableMaxSize)
{
    unsigned int pos = dict_hash_table_hash_str(skey) % hashTableMaxSize;
    DictHashNode *pHead = *(dictHashTable + pos);

    while (pHead)
    {
        if (wcscmp(skey, pHead->sKey) == 0)
            return pHead->nValue;

        pHead = pHead->pNext;
    }
    return -1;
}

// free the memory of the hash table
void dict_hash_table_release(DictHashNode **dictHashTable, int hashTableMaxSize)
{
    int i;
    for (i = 0; i < hashTableMaxSize; ++i)
    {
        if (*(dictHashTable + i))
        {
            DictHashNode *pHead = *(dictHashTable + i);
            while (pHead)
            {
                DictHashNode *pTemp = pHead;
                pHead = pHead->pNext;
                if (pTemp)
                {
                    free(pTemp->sKey);
                    free(pTemp);
                }
            }
        }
    }
}

/* ===============================hash table end=========================*/

float dict_detect_empty_bucket(int count_table[], int hashTableMaxSize)
{
    int cnt = 0;
    for (int i = 0; i < hashTableMaxSize; i++)
    {
        if (count_table[i] == 0)
        {
            cnt++;
        }
    }

    float empty_ratio = (float)cnt / hashTableMaxSize;
    return empty_ratio;
}

double dict_kl_divergence(int count_table[], int hashTableMaxSize, int hashTableSize)
{
    double pre_prob_log = log((double)hashTableMaxSize);
    double cross_entropy = 0.0, entropy = 0.0;

    for (int i = 0; i < hashTableMaxSize; i++)
    {
        double post_prob = (double)count_table[i] / hashTableSize;

        if (post_prob != 0)
        {
            cross_entropy += post_prob * pre_prob_log;
            entropy += post_prob * log(1 / post_prob);
        }
    }
    double kl_d = cross_entropy - entropy;
    printf("\tkl divergence: %f, diff_ratio: %f\n", kl_d, kl_d / entropy);
    return cross_entropy - entropy; // kl-d value
}

void dict_distribution_statistics(DictHashNode **dictHashTable, int hashTableMaxSize, int hashTableSize)
{
    printf("===========content of hash table=================\n");
    int countTable[hashTableMaxSize];
    float averageNum = (float)hashTableSize / hashTableMaxSize;

    for (int i = 0; i < hashTableMaxSize; ++i)
    {
        if (*(dictHashTable + i))
        {
            DictHashNode *pHead = *(dictHashTable + i);
            int k = 0;
            while (pHead)
            {
                pHead = pHead->pNext;
                k++;
            }
            countTable[i] = k;
        }
        else
        {
            countTable[i] = 0;
        }
    }

    double kl_d = dict_kl_divergence(countTable, hashTableMaxSize, hashTableSize);

    float empty_ratio = dict_detect_empty_bucket(countTable, hashTableMaxSize);
    printf("\tempty bucket ratio: %f\n", empty_ratio);
}
